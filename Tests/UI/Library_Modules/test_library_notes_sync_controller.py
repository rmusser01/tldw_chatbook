"""Controller tests against a recording lasting-sync runtime port."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from inspect import signature

import pytest

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncApplyBlocker,
)
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
    RuntimeConflictLabel,
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
        self.labels: tuple[RuntimeConflictLabel, ...] | None = None

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

    async def conflict_labels(
        self, root_id: str, token: str
    ) -> tuple[RuntimeConflictLabel, ...]:
        self.calls.append(("conflict_labels", root_id, token))
        if self.labels is not None:
            return self.labels
        plans = tuple(
            plan
            for plan in (
                self.check_plan,
                self.apply_result.fresh_plan if self.apply_result is not None else None,
            )
            if plan is not None and plan.observation_token == token
        )
        if not plans:
            return ()
        plan = plans[-1]
        return tuple(
            RuntimeConflictLabel(
                attention.binding_id,
                f"Title {attention.binding_id}",
                f"{attention.binding_id}.md",
            )
            for attention in plan.attention
            if attention.kind is ReconciliationAttentionKind.CONFLICT
            and attention.binding_id is not None
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

    async def conflict_history_available(self, root_id: str) -> bool:
        self.calls.append(("conflict_history_available", root_id))
        return bool(self.history.get(0, ()))

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
    await controller.apply_reviewed("root-1", TOKEN)

    assert runtime.calls == [
        ("check_root", "root-1"),
        ("conflict_labels", "root-1", TOKEN),
        ("conflict_history_available", "root-1"),
        ("apply_reviewed", "root-1", TOKEN, ("act-1",), ()),
        ("active_conflict_receipts", "root-1"),
        ("conflict_labels", "root-1", TOKEN_2),
        ("conflict_history_available", "root-1"),
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
        ("conflict_labels", root_id, TOKEN),
        ("conflict_history_available", root_id),
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

    await controller.apply_reviewed("root-1", TOKEN)

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
    await controller.apply_reviewed("root-1", TOKEN)

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
    assert controller.snapshot.review.stale is True
    assert (
        controller.snapshot.review.apply_blocker is LastingSyncApplyBlocker.STALE_REVIEW
    )
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

    await controller.apply_reviewed("root-1", TOKEN)

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

    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")

    assert tuple(runtime.calls) == calls_before
    assert publications == ["Choice staged. No changes yet."]
    assert controller.snapshot.review.rows[0].selected_choice is (
        NotesSyncConflictChoice.KEEP_BOTH
    )
    controller.set_review_page(2)
    assert controller.snapshot.review.rows[0].item_id == "bind-2"
    controller.set_review_page(1)
    assert controller.snapshot.review.rows[0].selected_label == "Selected: Keep both"


async def test_check_projects_exact_collapsed_labels_and_durable_history_availability() -> (
    None
):
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    runtime.labels = (
        RuntimeConflictLabel("bind-1", "Release note", "notes/release.md"),
    )
    runtime.history[0] = (
        RuntimeConflictHistoryRow(
            "operation-1",
            "Release note · notes/release.md",
            NotesSyncConflictChoice.KEEP_FILE,
            "completed",
            "2026-08-22T12:00:00+00:00",
            "2026-08-22T12:00:00+00:00",
            True,
        ),
    )
    controller = LibraryNotesSyncController(
        runtime=runtime, import_controller=_ImportController()
    )

    await controller.check_root("root-1")

    row = controller.snapshot.review.rows[0]
    assert (row.conflict_title, row.conflict_relative_path) == (
        "Release note",
        "notes/release.md",
    )
    assert controller.snapshot.history_available is True
    assert ("conflict_labels", "root-1", TOKEN) in runtime.calls
    assert ("conflict_history_available", "root-1") in runtime.calls


async def test_failed_review_fact_projection_disables_history_without_calling_compare() -> (
    None
):
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()

    async def unavailable(*_args: object, **_kwargs: object):
        raise RuntimeError("unavailable")

    runtime.conflict_labels = unavailable
    runtime.conflict_history_available = unavailable
    controller = LibraryNotesSyncController(
        runtime=runtime, import_controller=_ImportController()
    )

    await controller.check_root("root-1")

    assert controller.snapshot.review.rows[0].conflict_title == ""
    assert controller.snapshot.history_available is False
    assert not any(call[0] == "compare_conflict" for call in runtime.calls)


@pytest.mark.parametrize(
    "labels",
    (
        (),
        (
            RuntimeConflictLabel("bind-1", "One", "one.md"),
            RuntimeConflictLabel("bind-1", "Duplicate", "duplicate.md"),
        ),
    ),
)
async def test_incomplete_or_duplicate_labels_make_review_non_actionable(
    labels: tuple[RuntimeConflictLabel, ...],
) -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    runtime.labels = labels
    controller = LibraryNotesSyncController(
        runtime=runtime, import_controller=_ImportController()
    )

    await controller.check_root("root-1")

    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.can_apply is False
    assert controller.snapshot.review.rows[0].conflict_title == ""
    assert "unavailable" in controller.snapshot.status_line


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
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    assert controller._selections == {  # noqa: SLF001 - token key is the contract
        (TOKEN, "bind-1"): NotesSyncConflictChoice.KEEP_FILE
    }

    runtime.check_plan = _conflict_plan(token=TOKEN_2)
    await controller.check_root("root-1")
    assert controller.snapshot.review.rows[0].selected_choice is None

    controller.stage_attention_choice("root-1", TOKEN_2, "bind-1", "Keep note")
    runtime.stale = True
    await controller.apply_reviewed("root-1", TOKEN_2)
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.rows[0].selected_choice is None

    runtime.stale = False
    runtime.check_plan = _conflict_plan(root_id="root-2", token=TOKEN)
    await controller.check_root("root-2")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")
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
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")

    await controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
    assert controller.snapshot.comparison is not None
    controller.return_to_conflict_choices("root-1", TOKEN, "bind-1")
    assert controller.snapshot.comparison is None
    assert controller.snapshot.review.rows[0].selected_choice is (
        NotesSyncConflictChoice.KEEP_FILE
    )

    await controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
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

    pending = asyncio.create_task(
        controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
    )
    await first_started.wait()
    controller.invalidate_for_remount()
    await controller.check_root("root-1")
    current = asyncio.create_task(
        controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
    )
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
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep note")

    await controller.apply_reviewed("root-1", TOKEN)

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
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    statuses.clear()

    await controller.apply_reviewed("root-1", TOKEN)

    assert controller.snapshot.phase == "review"
    assert controller.snapshot.review.observation_token == TOKEN_2
    assert len(controller.snapshot.receipts) == 1
    assert statuses == ["2 applied · 1 conflict remains."]
    assert controller.snapshot.conflict_focus_binding_id == "bind-1"


async def test_fresh_fact_failure_clears_applied_selection_and_comparison() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    result = NotesSyncExecutionResult(
        "operation-1", NotesSyncOperationState.COMPLETED, False
    )
    runtime.apply_result = ConflictApplyResult(
        (result,),
        0,
        1,
        1,
        True,
        False,
        False,
        _conflict_plan(token=TOKEN_2),
    )

    async def labels(root_id: str, token: str) -> tuple[RuntimeConflictLabel, ...]:
        if token == TOKEN_2:
            raise RuntimeError("fresh labels unavailable")
        return (RuntimeConflictLabel("bind-1", "Note", "note.md"),)

    runtime.conflict_labels = labels
    controller = LibraryNotesSyncController(
        runtime=runtime, import_controller=_ImportController()
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    await controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
    assert controller.snapshot.comparison is not None

    await controller.apply_reviewed("root-1", TOKEN)

    assert controller._selections == {}
    assert controller.snapshot.comparison is None
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.can_apply is False


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
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")

    await controller.apply_reviewed("root-1", TOKEN)

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


def _receipt(operation_id: str, label: str) -> RuntimeConflictReceipt:
    return RuntimeConflictReceipt(
        operation_id,
        label,
        NotesSyncConflictChoice.KEEP_FILE,
        "completed",
        True,
    )


def _history_row(
    operation_id: str,
    label: str,
    *,
    state: str = "completed",
    undo_available: bool = True,
    undo_reason: str | None = None,
) -> RuntimeConflictHistoryRow:
    return RuntimeConflictHistoryRow(
        operation_id,
        label,
        NotesSyncConflictChoice.KEEP_FILE,
        state,
        "2026-08-22T12:00:00+00:00",
        "2026-08-22T12:00:00+00:00",
        undo_available,
        undo_reason,
    )


async def test_root_switch_clears_and_fences_delayed_receipt_and_history_results() -> (
    None
):
    runtime = _Runtime()
    runtime.receipts = (_receipt("operation-old", "Old root"),)
    runtime.history[0] = (_history_row("operation-old", "Old root"),)
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.refresh_conflict_receipts("root-1")
    await controller.show_resolution_history("root-1")
    assert controller.snapshot.receipts
    assert controller.snapshot.history.rows

    receipt_started = {root: asyncio.Event() for root in ("root-1", "root-2")}
    history_started = {root: asyncio.Event() for root in ("root-1", "root-2")}
    receipt_release = {root: asyncio.Event() for root in ("root-1", "root-2")}
    history_release = {root: asyncio.Event() for root in ("root-1", "root-2")}

    async def delayed_receipts(root_id: str) -> tuple[RuntimeConflictReceipt, ...]:
        receipt_started[root_id].set()
        await receipt_release[root_id].wait()
        return (_receipt(f"operation-{root_id}", root_id),)

    async def delayed_history(
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        history_started[root_id].set()
        await history_release[root_id].wait()
        return (_history_row(f"operation-{root_id}", root_id),)

    runtime.active_conflict_receipts = delayed_receipts
    runtime.resolution_history = delayed_history
    old_receipts = asyncio.create_task(controller.refresh_conflict_receipts("root-1"))
    old_history = asyncio.create_task(controller.show_resolution_history("root-1"))
    await receipt_started["root-1"].wait()
    await history_started["root-1"].wait()

    runtime.check_plan = _conflict_plan(root_id="root-2")
    await controller.check_root("root-2")
    assert controller.snapshot.receipts == ()
    assert controller.snapshot.history.rows == ()

    new_receipts = asyncio.create_task(controller.refresh_conflict_receipts("root-2"))
    new_history = asyncio.create_task(controller.show_resolution_history("root-2"))
    await receipt_started["root-2"].wait()
    await history_started["root-2"].wait()
    receipt_release["root-2"].set()
    history_release["root-2"].set()
    await asyncio.gather(new_receipts, new_history)

    receipt_release["root-1"].set()
    history_release["root-1"].set()
    await asyncio.gather(old_receipts, old_history)
    assert controller.snapshot.receipts[0].item_label == "root-2"
    assert controller.snapshot.history.root_id == "root-2"
    assert controller.snapshot.history.rows[0].item_label == "root-2"


async def test_remount_and_history_page_generations_reject_out_of_order_results() -> (
    None
):
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    started = [asyncio.Event() for _ in range(4)]
    release = [asyncio.Event() for _ in range(4)]
    calls = 0

    async def delayed_history(
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        nonlocal calls
        if limit == 1:
            return ()
        call = calls
        calls += 1
        started[call].set()
        await release[call].wait()
        return (_history_row(f"operation-{call}", f"page-{offset // 100 + 1}"),)

    runtime.resolution_history = delayed_history
    page_one = asyncio.create_task(controller.show_resolution_history("root-1", page=1))
    await started[0].wait()
    page_two = asyncio.create_task(controller.show_resolution_history("root-1", page=2))
    await started[1].wait()
    release[1].set()
    await page_two
    release[0].set()
    await page_one
    assert controller.snapshot.history.page == 2
    assert controller.snapshot.history.rows[0].item_label == "page-2"

    old = asyncio.create_task(controller.show_resolution_history("root-1", page=2))
    await started[2].wait()
    controller.invalidate_for_remount()
    fresh = asyncio.create_task(controller.show_resolution_history("root-1", page=2))
    await started[3].wait()
    release[3].set()
    await fresh
    release[2].set()
    await old
    assert controller.snapshot.history.rows[0].operation_id == "operation-3"


async def test_remount_generation_rejects_stale_same_root_receipt_completion() -> None:
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def delayed_receipts(root_id: str) -> tuple[RuntimeConflictReceipt, ...]:
        nonlocal calls
        call = calls
        calls += 1
        started[call].set()
        await release[call].wait()
        return (_receipt(f"operation-{call}", "Old" if call == 0 else "Fresh"),)

    runtime.active_conflict_receipts = delayed_receipts
    old = asyncio.create_task(controller.refresh_conflict_receipts("root-1"))
    await started[0].wait()
    controller.invalidate_for_remount()
    fresh = asyncio.create_task(controller.refresh_conflict_receipts("root-1"))
    await started[1].wait()
    release[1].set()
    await fresh
    release[0].set()
    await old

    assert controller.snapshot.receipts[0].item_label == "Fresh"


async def test_completed_undo_refreshes_open_history_once_and_nonterminal_recovers() -> (
    None
):
    runtime = _Runtime()
    runtime.receipts = (_receipt("operation-1", "Note"),)
    runtime.history[0] = (_history_row("operation-1", "Note"),)
    statuses: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: statuses.append(snapshot.status_line),
    )
    await controller.show_resolution_history("root-1", page=1)

    async def completed_undo(
        root_id: str, operation_id: str
    ) -> NotesSyncExecutionResult:
        runtime.receipts = ()
        runtime.history[0] = (
            _history_row(
                operation_id,
                "Note",
                state="undone",
                undo_available=False,
                undo_reason="Undone",
            ),
        )
        return NotesSyncExecutionResult(
            "undo-operation-1", NotesSyncOperationState.COMPLETED, False
        )

    runtime.undo_resolution = completed_undo
    statuses.clear()
    await controller.undo_conflict_resolution("root-1", "operation-1")
    assert statuses == ["Undo finished. Check changes before applying again."]
    assert controller.snapshot.phase == "history"
    assert controller.snapshot.history.rows[0].state == "undone"
    assert controller.snapshot.receipts == ()

    from tldw_chatbook.Notes.notes_sync_executor import NotesSyncRecoveryChoice

    async def nonterminal_undo(
        root_id: str, operation_id: str
    ) -> NotesSyncExecutionResult:
        return NotesSyncExecutionResult(
            "undo-operation-2",
            NotesSyncOperationState.NEEDS_ATTENTION,
            True,
            "operation_failed",
            tuple(NotesSyncRecoveryChoice),
        )

    runtime.undo_resolution = nonterminal_undo
    statuses.clear()
    await controller.undo_conflict_resolution("root-1", "operation-1")
    assert controller.snapshot.phase == "roots"
    assert "recovery needs attention" in controller.snapshot.status_line.casefold()
    assert "Undo finished" not in controller.snapshot.status_line
    assert statuses == [controller.snapshot.status_line]

    async def intermediate_undo(
        root_id: str, operation_id: str
    ) -> NotesSyncExecutionResult:
        return NotesSyncExecutionResult(
            "undo-operation-3", NotesSyncOperationState.VERIFIED, False
        )

    runtime.undo_resolution = intermediate_undo
    statuses.clear()
    await controller.undo_conflict_resolution("root-1", "operation-1")
    assert controller.snapshot.phase == "roots"
    assert "recovery needs attention" in controller.snapshot.status_line.casefold()
    assert "Undo finished" not in controller.snapshot.status_line
    assert statuses == [controller.snapshot.status_line]


async def test_delayed_completed_undo_cannot_overwrite_newer_history_page() -> None:
    runtime = _Runtime()
    runtime.receipts = (_receipt("operation-1", "Page one"),)
    runtime.history[0] = (_history_row("operation-1", "Page one"),)
    statuses: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: statuses.append(snapshot.status_line),
    )
    await controller.refresh_conflict_receipts("root-1")
    await controller.show_resolution_history("root-1", page=1)
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def completed_undo(
        root_id: str, operation_id: str
    ) -> NotesSyncExecutionResult:
        runtime.receipts = ()
        return NotesSyncExecutionResult(
            "undo-operation-1", NotesSyncOperationState.COMPLETED, False
        )

    async def delayed_history(
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        if offset == 0:
            refresh_started.set()
            await release_refresh.wait()
            return (_history_row("operation-stale", "Stale page one"),)
        return (_history_row("operation-current", "Page two"),)

    runtime.undo_resolution = completed_undo
    runtime.resolution_history = delayed_history
    pending = asyncio.create_task(
        controller.undo_conflict_resolution("root-1", "operation-1")
    )
    await refresh_started.wait()
    await controller.show_resolution_history("root-1", page=2)
    assert controller.snapshot.receipts[0].operation_id == "operation-1"
    statuses.clear()
    release_refresh.set()
    await pending

    assert controller.snapshot.history.page == 2
    assert controller.snapshot.history.rows[0].operation_id == "operation-current"
    assert controller.snapshot.receipts == ()
    assert controller.snapshot.status_line == "Resolution history loaded."
    assert statuses == ["Resolution history loaded."]


async def test_history_page_bounds_reject_before_runtime_call() -> None:
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    largest_page = ((2**63 - 1) // 100) + 1

    for invalid in (True, largest_page + 1, 10**100):
        with pytest.raises(ValueError, match="history page|SQLite"):
            await controller.show_resolution_history("root-1", page=invalid)
    assert not any(call[0] == "resolution_history" for call in runtime.calls)


async def test_out_of_order_root_checks_and_remounted_error_publish_nothing_stale() -> (
    None
):
    runtime = _Runtime()
    started = {root: asyncio.Event() for root in ("root-1", "root-2")}
    release = {root: asyncio.Event() for root in ("root-1", "root-2")}

    async def delayed_check(root_id: str) -> ReconciliationPlan:
        started[root_id].set()
        await release[root_id].wait()
        return _conflict_plan(
            root_id=root_id,
            token=TOKEN if root_id == "root-1" else TOKEN_2,
        )

    runtime.check_root = delayed_check
    published: list[object] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=published.append,
    )
    first = asyncio.create_task(controller.check_root("root-1"))
    await started["root-1"].wait()
    second = asyncio.create_task(controller.check_root("root-2"))
    await started["root-2"].wait()
    release["root-2"].set()
    await second
    count = len(published)
    release["root-1"].set()
    await first
    assert len(published) == count
    assert controller.snapshot.review.root_id == "root-2"
    assert controller.snapshot.review.observation_token == TOKEN_2

    error_started = asyncio.Event()
    release_error = asyncio.Event()

    async def delayed_error(root_id: str) -> ReconciliationPlan:
        error_started.set()
        await release_error.wait()
        raise RuntimeError("old failure")

    runtime.check_root = delayed_error
    pending = asyncio.create_task(controller.check_root("root-2"))
    await error_started.wait()
    controller.invalidate_for_remount()
    count = len(published)
    release_error.set()
    await pending
    assert len(published) == count


@pytest.mark.parametrize("delayed_fact", ("labels", "history"))
async def test_superseded_review_facts_cannot_clobber_new_root_projection(
    delayed_fact: str,
) -> None:
    runtime = _Runtime()
    facts_started = asyncio.Event()
    release_facts = asyncio.Event()

    async def check(root_id: str) -> ReconciliationPlan:
        return _conflict_plan(
            root_id=root_id,
            token=TOKEN if root_id == "root-1" else TOKEN_2,
        )

    async def labels(root_id: str, token: str) -> tuple[RuntimeConflictLabel, ...]:
        if root_id == "root-1" and delayed_fact == "labels":
            facts_started.set()
            await release_facts.wait()
        return (RuntimeConflictLabel("bind-1", root_id, f"{token}.md"),)

    async def history_available(root_id: str) -> bool:
        if root_id == "root-1" and delayed_fact == "history":
            facts_started.set()
            await release_facts.wait()
        return root_id == "root-2"

    runtime.check_root = check
    runtime.conflict_labels = labels
    runtime.conflict_history_available = history_available
    published: list[object] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=published.append,
    )

    old = asyncio.create_task(controller.check_root("root-1"))
    await facts_started.wait()
    await controller.check_root("root-2")
    expected = controller.snapshot
    expected_plan = controller._review_plan
    expected_labels = controller._review_labels.copy()
    count = len(published)

    release_facts.set()
    await old

    assert controller.snapshot == expected
    assert controller._review_plan is expected_plan
    assert controller._review_labels == expected_labels
    assert controller._projection_root_id == "root-2"
    assert len(published) == count


async def test_superseded_same_root_review_facts_keep_newer_token() -> None:
    runtime = _Runtime()
    labels_started = asyncio.Event()
    release_labels = asyncio.Event()
    check_count = 0

    async def check(root_id: str) -> ReconciliationPlan:
        nonlocal check_count
        check_count += 1
        return _conflict_plan(token=TOKEN if check_count == 1 else TOKEN_2)

    async def labels(root_id: str, token: str) -> tuple[RuntimeConflictLabel, ...]:
        if token == TOKEN:
            labels_started.set()
            await release_labels.wait()
        return (RuntimeConflictLabel("bind-1", token, "note.md"),)

    runtime.check_root = check
    runtime.conflict_labels = labels
    controller = LibraryNotesSyncController(
        runtime=runtime, import_controller=_ImportController()
    )

    old = asyncio.create_task(controller.check_root("root-1"))
    await labels_started.wait()
    await controller.check_root("root-1")
    expected = controller.snapshot
    release_labels.set()
    await old

    assert controller.snapshot == expected
    assert controller.snapshot.review.observation_token == TOKEN_2
    assert controller._review_plan is not None
    assert controller._review_plan.observation_token == TOKEN_2


async def test_remount_fences_delayed_review_facts_without_publication() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    labels_started = asyncio.Event()
    release_labels = asyncio.Event()

    async def delayed_labels(
        root_id: str, token: str
    ) -> tuple[RuntimeConflictLabel, ...]:
        labels_started.set()
        await release_labels.wait()
        return (RuntimeConflictLabel("bind-1", "Note", "note.md"),)

    runtime.conflict_labels = delayed_labels
    published: list[object] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=published.append,
    )
    pending = asyncio.create_task(controller.check_root("root-1"))
    await labels_started.wait()
    controller.invalidate_for_remount()
    expected = controller.snapshot
    count = len(published)

    release_labels.set()
    await pending

    assert controller.snapshot == expected
    assert controller._projection_root_id is None
    assert controller._review_plan is None
    assert controller._review_labels == {}
    assert len(published) == count


async def test_failed_superseded_review_facts_publish_nothing() -> None:
    runtime = _Runtime()
    labels_started = asyncio.Event()
    release_labels = asyncio.Event()

    async def check(root_id: str) -> ReconciliationPlan:
        return _conflict_plan(
            root_id=root_id,
            token=TOKEN if root_id == "root-1" else TOKEN_2,
        )

    async def labels(root_id: str, token: str) -> tuple[RuntimeConflictLabel, ...]:
        if root_id == "root-1":
            labels_started.set()
            await release_labels.wait()
            raise RuntimeError("old labels unavailable")
        return (RuntimeConflictLabel("bind-1", "Current", "current.md"),)

    runtime.check_root = check
    runtime.conflict_labels = labels
    published: list[object] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=published.append,
    )
    old = asyncio.create_task(controller.check_root("root-1"))
    await labels_started.wait()
    await controller.check_root("root-2")
    expected = controller.snapshot
    count = len(published)

    release_labels.set()
    await old

    assert controller.snapshot == expected
    assert controller._review_plan is not None
    assert controller._review_plan.root_id == "root-2"
    assert len(published) == count


async def test_same_root_receipt_refresh_does_not_drop_apply_or_undo_result() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    apply_started = asyncio.Event()
    release_apply = asyncio.Event()
    original_apply = runtime.apply_reviewed

    async def delayed_apply(*args, **kwargs) -> ConflictApplyResult:
        apply_started.set()
        await release_apply.wait()
        return await original_apply(*args, **kwargs)

    runtime.apply_reviewed = delayed_apply
    pending_apply = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    await apply_started.wait()
    await controller.refresh_conflict_receipts("root-1")
    release_apply.set()
    await pending_apply
    assert controller.snapshot.phase == "receipt"
    assert "no conflicts remain" in controller.snapshot.status_line

    receipt = _receipt("operation-1", "Note")
    runtime.receipts = (receipt,)
    await controller.refresh_conflict_receipts("root-1")
    undo_started = asyncio.Event()
    release_undo = asyncio.Event()

    async def delayed_undo(root_id: str, operation_id: str) -> NotesSyncExecutionResult:
        undo_started.set()
        await release_undo.wait()
        runtime.receipts = ()
        return NotesSyncExecutionResult(
            "undo-operation-1", NotesSyncOperationState.COMPLETED, False
        )

    runtime.undo_resolution = delayed_undo
    pending_undo = asyncio.create_task(
        controller.undo_conflict_resolution("root-1", "operation-1")
    )
    await undo_started.wait()
    await controller.refresh_conflict_receipts("root-1")
    release_undo.set()
    await pending_undo
    assert controller.snapshot.receipts == ()
    assert controller.snapshot.status_line.startswith("Undo finished")


async def test_new_same_root_check_supersedes_older_apply_view() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    started = asyncio.Event()
    release = asyncio.Event()
    original_apply = runtime.apply_reviewed

    async def delayed_apply(*args, **kwargs) -> ConflictApplyResult:
        started.set()
        await release.wait()
        return await original_apply(*args, **kwargs)

    runtime.apply_reviewed = delayed_apply
    pending = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    await started.wait()
    runtime.check_plan = _conflict_plan(token=TOKEN_2)
    await controller.check_root("root-1")
    release.set()
    await pending

    assert controller.snapshot.phase == "review"
    assert controller.snapshot.review.observation_token == TOKEN_2


async def test_successful_mutations_use_safe_local_fallback_when_receipts_fail() -> (
    None
):
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    statuses: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: statuses.append(snapshot.status_line),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")

    async def receipt_failure(root_id: str) -> tuple[RuntimeConflictReceipt, ...]:
        raise RuntimeError("projection unavailable")

    runtime.active_conflict_receipts = receipt_failure
    statuses.clear()
    await controller.apply_reviewed("root-1", TOKEN)
    assert controller.snapshot.receipts_unavailable is True
    assert "receipt" in controller.snapshot.status_line.casefold()
    assert statuses == [controller.snapshot.status_line]

    runtime = _Runtime()
    runtime.receipts = (_receipt("operation-1", "Note"),)
    runtime.history[0] = (_history_row("operation-1", "Note"),)
    statuses = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: statuses.append(snapshot.status_line),
    )
    await controller.refresh_conflict_receipts("root-1")
    await controller.show_resolution_history("root-1")
    history_before_dismiss = controller.snapshot.history
    runtime.active_conflict_receipts = receipt_failure
    statuses.clear()
    await controller.dismiss_conflict_receipt("root-1", "operation-1")
    assert controller.snapshot.receipts == ()
    assert controller.snapshot.receipts_unavailable is True
    assert controller.snapshot.history == history_before_dismiss
    assert controller.snapshot.history.rows[0].undo_available is True
    assert statuses == [controller.snapshot.status_line]

    await controller.show_resolution_history("root-1")
    assert controller.snapshot.history.rows[0].undo_available is True

    runtime.receipts = (_receipt("operation-1", "Note"),)

    async def receipt_success(root_id: str) -> tuple[RuntimeConflictReceipt, ...]:
        return runtime.receipts

    runtime.active_conflict_receipts = receipt_success
    await controller.refresh_conflict_receipts("root-1")
    runtime.active_conflict_receipts = receipt_failure
    statuses.clear()
    await controller.undo_conflict_resolution("root-1", "operation-1")
    assert controller.snapshot.receipts == ()
    assert controller.snapshot.receipts_unavailable is True
    assert controller.snapshot.history.rows[0].state == "undone"
    assert controller.snapshot.history.rows[0].undo_available is False
    assert statuses == [controller.snapshot.status_line]


async def test_root_activation_clears_choices_and_fences_comparison_on_receipt_switch() -> (
    None
):
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    gate = asyncio.Event()
    runtime.comparison_gate = gate
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    pending = asyncio.create_task(
        controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
    )
    await runtime.comparison_started.wait()

    await controller.refresh_conflict_receipts("root-2")
    assert controller._selections == {}  # noqa: SLF001 - lifecycle contract
    assert controller.snapshot.comparison is None
    controller._expanded_binding_id = "bind-1"  # noqa: SLF001 - isolate root fence
    assert not controller._comparison_is_current(  # noqa: SLF001
        controller._lifecycle_epoch,  # noqa: SLF001
        controller._comparison_generation,  # noqa: SLF001
        "root-1",
        TOKEN,
        "bind-1",
    )
    gate.set()
    await pending
    assert controller.snapshot.comparison is None


async def test_invalid_stage_and_comparison_publish_bounded_status_once() -> None:
    runtime = _Runtime()
    statuses: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: statuses.append(snapshot.status_line),
    )
    statuses.clear()
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    assert len(statuses) == 1
    assert (
        "unavailable" in statuses[0].casefold() or "invalid" in statuses[0].casefold()
    )

    runtime.check_plan = _conflict_plan()
    await controller.check_root("root-1")

    async def mismatch(root_id: str, token: str, binding_id: str) -> ConflictComparison:
        raise ValueError("comparison_binding_mismatch")

    runtime.compare_conflict = mismatch
    statuses.clear()
    await controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
    assert len(statuses) == 1
    assert "unavailable" in statuses[0].casefold()


@pytest.mark.parametrize("malformed", [object(), {"binding_id": "bind-1"}])
async def test_malformed_comparison_result_publishes_bounded_status_once(
    malformed: object,
) -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    statuses: list[str] = []
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
        publish_snapshot=lambda snapshot: statuses.append(snapshot.status_line),
    )
    await controller.check_root("root-1")

    async def malformed_comparison(root_id: str, token: str, binding_id: str) -> object:
        return malformed

    runtime.compare_conflict = malformed_comparison
    statuses.clear()
    await controller.show_conflict_comparison("root-1", TOKEN, "bind-1")

    assert statuses == [controller.snapshot.status_line]
    assert "comparison unavailable" in controller.snapshot.status_line.casefold()
    assert controller.snapshot.comparison is None


async def test_new_check_supersedes_pending_activate_and_control_publication() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    activate_started = asyncio.Event()
    release_activate = asyncio.Event()

    async def delayed_activate(
        root_id: str, authorization: object
    ) -> NotesSyncControlResult:
        activate_started.set()
        await release_activate.wait()
        return NotesSyncControlResult(True, "up_to_date", "sync_now")

    runtime.activate_root = delayed_activate
    pending_activate = asyncio.create_task(controller.activate_root("root-1"))
    await activate_started.wait()
    runtime.check_plan = _conflict_plan(token=TOKEN_2)
    await controller.check_root("root-1")
    release_activate.set()
    await pending_activate
    assert controller.snapshot.phase == "review"
    assert controller.snapshot.review.observation_token == TOKEN_2

    control_started = asyncio.Event()
    release_control = asyncio.Event()

    async def delayed_pause(root_id: str) -> NotesSyncControlResult:
        control_started.set()
        await release_control.wait()
        return NotesSyncControlResult(True, "paused", "resume_sync")

    runtime.pause_root = delayed_pause
    pending_control = asyncio.create_task(controller.pause_root("root-1"))
    await control_started.wait()
    runtime.check_plan = _conflict_plan(token=TOKEN)
    await controller.check_root("root-1")
    release_control.set()
    await pending_control
    assert controller.snapshot.phase == "review"
    assert controller.snapshot.review.observation_token == TOKEN


@pytest.mark.parametrize("failure", ["exception", "wrong_root", "malformed"])
async def test_failed_recheck_invalidates_prior_review_authority(
    failure: str,
) -> None:
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    assert controller.snapshot.review.can_apply is True

    async def failed_check(root_id: str) -> object:
        if failure == "exception":
            raise RuntimeError("check unavailable")
        if failure == "wrong_root":
            return _conflict_plan(root_id="root-2")
        return {"root_id": root_id, "observation_token": TOKEN_2}

    runtime.check_root = failed_check
    await controller.check_root("root-1")

    assert controller._review_plan is None  # noqa: SLF001 - authority contract
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.can_apply is False
    calls_before_apply = tuple(runtime.calls)
    await controller.apply_reviewed("root-1", TOKEN)
    assert tuple(runtime.calls) == calls_before_apply


@pytest.mark.parametrize(
    "operation", ["sync_now", "check_setup", "activate", "control"]
)
async def test_failed_fresh_review_paths_clear_prior_authority(operation: str) -> None:
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    assert controller.snapshot.review.can_apply is True

    async def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("operation unavailable")

    if operation == "sync_now":
        runtime.request_sync_now = fail
        await controller.sync_now("root-1")
    elif operation == "check_setup":
        controller.set_setup("display_name", "Notes")
        controller.set_setup("folder", "/private/root")
        runtime.review_setup = fail
        await controller.check_setup()
    elif operation == "activate":
        runtime.activate_root = fail
        await controller.activate_root("root-1")
    else:
        runtime.pause_root = fail
        await controller.pause_root("root-1")

    assert controller._review_plan is None  # noqa: SLF001 - authority contract
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.can_apply is False
    calls_before_apply = tuple(runtime.calls)
    await controller.apply_reviewed("root-1", TOKEN)
    assert tuple(runtime.calls) == calls_before_apply


async def _invoke_root_control(
    controller: LibraryNotesSyncController, operation: str
) -> None:
    if operation == "activate":
        await controller.activate_root("root-1")
    elif operation == "pause":
        await controller.pause_root("root-1")
    elif operation == "resume":
        await controller.resume_root("root-1")
    elif operation == "retarget":
        await controller.retarget_root("root-1", "/private/new-root")
    elif operation == "disconnect":
        await controller.disconnect_root("root-1", keep_folder_organization=True)
    elif operation == "cleanup":
        await controller.resolve_cleanup("root-1", "operation-1")
    else:
        controller.stage_root_action("root-1", "recover")


@pytest.mark.parametrize(
    "operation",
    ["activate", "pause", "resume", "retarget", "disconnect", "cleanup", "stage"],
)
@pytest.mark.parametrize("pending_kind", ["apply", "check"])
async def test_same_root_control_supersedes_pending_review_work(
    operation: str, pending_kind: str
) -> None:
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    started = asyncio.Event()
    release = asyncio.Event()

    if pending_kind == "apply":
        original_apply = runtime.apply_reviewed

        async def delayed_apply(*args: object, **kwargs: object) -> ConflictApplyResult:
            started.set()
            await release.wait()
            return await original_apply(*args, **kwargs)

        runtime.apply_reviewed = delayed_apply
        pending = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    else:

        async def delayed_check(root_id: str) -> ReconciliationPlan:
            started.set()
            await release.wait()
            return _conflict_plan(root_id=root_id, token=TOKEN_2)

        runtime.check_root = delayed_check
        pending = asyncio.create_task(controller.check_root("root-1"))

    await started.wait()
    await _invoke_root_control(controller, operation)
    control_snapshot = controller.snapshot
    release.set()
    await pending

    assert controller.snapshot == control_snapshot


async def test_unavailable_activation_supersedes_pending_same_root_check() -> None:
    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot(
        "awaiting_cutover", "finish_upgrade"
    )
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_check(root_id: str) -> ReconciliationPlan:
        started.set()
        await release.wait()
        return _conflict_plan(root_id=root_id, token=TOKEN_2)

    runtime.check_root = delayed_check
    pending = asyncio.create_task(controller.check_root("root-1"))
    await started.wait()
    assert await controller.activate_root("root-1") is False
    unavailable_snapshot = controller.snapshot
    release.set()
    await pending

    assert controller.snapshot == unavailable_snapshot
    assert controller._review_plan is None  # noqa: SLF001 - authority contract


async def test_same_root_controls_publish_only_the_newest_completion() -> None:
    runtime = _Runtime()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_retarget(root_id: str, target: str) -> NotesSyncControlResult:
        started.set()
        await release.wait()
        return NotesSyncControlResult(True, "up_to_date", "sync_now")

    runtime.retarget_root = delayed_retarget
    pending = asyncio.create_task(
        controller.retarget_root("root-1", "/private/new-root")
    )
    await started.wait()
    await controller.resume_root("root-1")
    newest_snapshot = controller.snapshot
    release.set()
    await pending

    assert controller.snapshot == newest_snapshot


async def test_undo_invalidates_active_review_but_preserves_receipt_history_context() -> (
    None
):
    runtime = _Runtime()
    runtime.receipts = (_receipt("operation-1", "Note"),)
    runtime.history[0] = (_history_row("operation-1", "Note"),)
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.refresh_conflict_receipts("root-1")
    await controller.show_resolution_history("root-1")
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    await controller.show_conflict_comparison("root-1", TOKEN, "bind-1")

    await controller.undo_conflict_resolution("root-1", "operation-1")

    assert controller._review_plan is None  # noqa: SLF001 - authority contract
    assert controller._selections == {}  # noqa: SLF001 - token authority contract
    assert controller.snapshot.review.stale is True
    assert controller.snapshot.review.can_apply is False
    assert controller.snapshot.comparison is None
    assert controller.snapshot.receipts == ()
    assert controller.snapshot.history.rows[0].state == "undone"
    assert controller.snapshot.history.rows[0].undo_available is False


@pytest.mark.parametrize("pending_kind", ["apply", "check"])
async def test_undo_supersedes_pending_same_root_review_work(
    pending_kind: str,
) -> None:
    runtime = _Runtime()
    runtime.receipts = (_receipt("operation-1", "Note"),)
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.refresh_conflict_receipts("root-1")
    await controller.check_root("root-1")
    started = asyncio.Event()
    release = asyncio.Event()

    if pending_kind == "apply":
        original_apply = runtime.apply_reviewed

        async def delayed_apply(*args: object, **kwargs: object) -> ConflictApplyResult:
            started.set()
            await release.wait()
            return await original_apply(*args, **kwargs)

        runtime.apply_reviewed = delayed_apply
        pending = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    else:

        async def delayed_check(root_id: str) -> ReconciliationPlan:
            started.set()
            await release.wait()
            return _conflict_plan(root_id=root_id, token=TOKEN_2)

        runtime.check_root = delayed_check
        pending = asyncio.create_task(controller.check_root("root-1"))

    await started.wait()
    await controller.undo_conflict_resolution("root-1", "operation-1")
    undo_snapshot = controller.snapshot
    release.set()
    await pending

    assert controller.snapshot == undo_snapshot
    assert controller.snapshot.receipts == ()


async def test_review_actions_reject_detached_review_provenance() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    await controller.show_conflict_comparison("root-1", TOKEN, "bind-1")
    comparison = controller.snapshot.comparison
    compare_calls = len(
        [call for call in runtime.calls if call[0] == "compare_conflict"]
    )

    controller.stage_attention_choice("root-1", TOKEN_2, "bind-1", "Keep note")
    await controller.show_conflict_comparison("root-1", TOKEN_2, "bind-1")
    controller.return_to_conflict_choices("root-1", TOKEN_2, "bind-1")
    await controller.apply_reviewed("root-1", TOKEN_2)

    assert controller.snapshot.comparison == comparison
    assert (
        controller._selections[(TOKEN, "bind-1")] is NotesSyncConflictChoice.KEEP_FILE
    )
    assert (TOKEN_2, "bind-1") not in controller._selections
    assert len([call for call in runtime.calls if call[0] == "compare_conflict"]) == (
        compare_calls
    )
    assert not any(call[0] == "apply_reviewed" for call in runtime.calls)


async def test_duplicate_in_flight_apply_provenance_invokes_runtime_once() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    started = asyncio.Event()
    release = asyncio.Event()
    original_apply = runtime.apply_reviewed
    invocations = 0

    async def delayed_apply(*args: object, **kwargs: object) -> ConflictApplyResult:
        nonlocal invocations
        invocations += 1
        started.set()
        await release.wait()
        return await original_apply(*args, **kwargs)

    runtime.apply_reviewed = delayed_apply
    first = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    await started.wait()
    second = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    await asyncio.sleep(0)
    assert invocations == 1

    release.set()
    await asyncio.gather(first, second)

    apply_calls = [call for call in runtime.calls if call[0] == "apply_reviewed"]
    assert len(apply_calls) == 1


async def test_duplicate_apply_stays_fenced_until_fresh_projection_finishes() -> None:
    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep file")
    receipts_started = asyncio.Event()
    release_receipts = asyncio.Event()
    original_receipts = runtime.active_conflict_receipts

    async def delayed_receipts(root_id: str) -> tuple[RuntimeConflictReceipt, ...]:
        receipts_started.set()
        await release_receipts.wait()
        return await original_receipts(root_id)

    runtime.active_conflict_receipts = delayed_receipts
    first = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    await receipts_started.wait()
    second = asyncio.create_task(controller.apply_reviewed("root-1", TOKEN))
    await asyncio.sleep(0)

    assert len([call for call in runtime.calls if call[0] == "apply_reviewed"]) == 1

    release_receipts.set()
    await asyncio.gather(first, second)


@pytest.mark.parametrize(
    ("row_count", "has_next"),
    ((99, False), (100, False), (101, True)),
)
async def test_history_has_next_uses_exact_sentinel(
    row_count: int, has_next: bool
) -> None:
    runtime = _Runtime()
    runtime.history[0] = tuple(
        _history_row(f"operation-{index}", f"Note {index}")
        for index in range(min(row_count, 100))
    )
    if row_count > 100:
        runtime.history[100] = (_history_row("operation-100", "Note 100"),)
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.show_resolution_history("root-1")

    assert len(controller.snapshot.history.rows) == min(row_count, 100)
    assert controller.snapshot.history.has_next is has_next
    assert ("resolution_history", "root-1", 1, 100) in runtime.calls


async def test_history_sentinel_failure_fails_closed() -> None:
    runtime = _Runtime()
    runtime.history[0] = (_history_row("operation-1", "Note"),)
    original = runtime.resolution_history

    async def failing_sentinel(
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        if limit == 1 and offset == 100:
            raise OSError("history unavailable")
        return await original(root_id, limit=limit, offset=offset, now=now)

    runtime.resolution_history = failing_sentinel
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    await controller.show_resolution_history("root-1")

    assert controller.snapshot.history.unavailable is True
    assert controller.snapshot.history.rows == ()
    assert controller.snapshot.history.has_next is False


async def test_stale_history_sentinel_cannot_overwrite_newer_page() -> None:
    runtime = _Runtime()
    sentinel_started = asyncio.Event()
    release_sentinel = asyncio.Event()

    async def delayed_sentinel(
        root_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        now: int | None = None,
    ) -> tuple[RuntimeConflictHistoryRow, ...]:
        if limit == 1 and offset == 100:
            sentinel_started.set()
            await release_sentinel.wait()
            return (_history_row("operation-old-sentinel", "Old sentinel"),)
        if limit == 100 and offset == 0:
            return (_history_row("operation-page-1", "Page one"),)
        if limit == 100 and offset == 100:
            return (_history_row("operation-page-2", "Page two"),)
        return ()

    runtime.resolution_history = delayed_sentinel
    controller = LibraryNotesSyncController(
        runtime=runtime,
        import_controller=_ImportController(),
    )

    old = asyncio.create_task(controller.show_resolution_history("root-1", page=1))
    await sentinel_started.wait()
    await controller.show_resolution_history("root-1", page=2)
    current = controller.snapshot
    release_sentinel.set()
    await old

    assert controller.snapshot == current
    assert controller.snapshot.history.page == 2
    assert controller.snapshot.history.rows[0].item_label == "Page two"
