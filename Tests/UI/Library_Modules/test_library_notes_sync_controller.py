"""Controller tests against a recording lasting-sync runtime port."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import pytest

from tldw_chatbook.Notes.notes_sync_models import NotesSyncAction, NotesSyncActionKind
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationPlan
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncControlResult,
    NotesSyncRootRuntimeSnapshot,
    NotesSyncRuntimeSnapshot,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)

pytestmark = pytest.mark.asyncio
TOKEN = "b" * 64


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

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        return NotesSyncRuntimeSnapshot(
            "active",
            "sync_now",
            (NotesSyncRootRuntimeSnapshot("root-1", "up_to_date", "sync_now"),),
        )

    async def check_root(self, root_id: str) -> ReconciliationPlan:
        self.calls.append(("check_root", root_id))
        return ReconciliationPlan(
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
        self, root_id: str, token: str, action_ids: tuple[str, ...]
    ) -> tuple[object, ...]:
        self.calls.append(("apply_reviewed", root_id, token, action_ids))
        if self.stale:
            raise ValueError("stale_review")
        return (object(),)

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
        ("apply_reviewed", "root-1", TOKEN, ("act-1",)),
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
