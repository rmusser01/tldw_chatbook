"""Slower arms of TASK-34000.16 (NOT in the UI PR gate census).

Two boots: the 100x30 repeat of the census arm, and an active-root Check from
Manage sync folders with a NEW latin-1 file dropped beside a real deletion --
the review names the skip AND the deletion side by side, and the root row never
paints the ``NotesSyncRootRefused`` class name the review's capture 03 showed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_notes_sync_skips_bad_files_by_name import (
    LATIN1_ROW,
    MIXED_ROW,
    _isolate_actor_pack_recovery,  # noqa: F401 -- autouse fixture
    _painted,
    build_sync_stack,
    open_keep_synced_review,
    write_review_fixture,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_selector,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

LEGACY_ROW = "legacy.md · Not UTF-8 text — convert it to UTF-8 or move it out of the folder"


def _painted_review_lines(app, screen) -> list[str]:
    """Each review row's line as painted inside its own box, unwrapped.

    task-32294's rule: a substring check against the whole screen encodes one
    terminal width's wrap points; cropping to the widget and collapsing the
    wrap asserts the WHOLE sentence inside the box that owns it.
    """

    strips = list(app.screen._compositor.render_strips())
    lines: list[str] = []
    for widget in screen.query(".notes-sync-review-line").results(Static):
        region = widget.region
        rows = [
            strips[y].crop(region.x, region.right).text
            for y in range(region.y, min(region.bottom, len(strips)))
        ]
        lines.append(" ".join(" ".join(rows).split()))
    return lines


async def _activate_from_review(screen, pilot, host, controller) -> None:
    await _wait_for_condition(
        pilot,
        lambda: "Skipped (2)" in _painted(host),
        message=lambda: (
            "review never painted 'Skipped (2)'; status line: "
            f"{controller.snapshot.status_line!r}"
        ),
    )
    activate = screen.query_one("#notes-sync-activate", Button)
    await _wait_for_condition(
        pilot,
        lambda: not activate.disabled,
        message=lambda: (
            f"review never admitted activation: {controller.snapshot.status_line!r}"
        ),
    )
    activate.press()
    await _wait_for_selector(screen, pilot, "#notes-sync-receipt")
    await _wait_for_condition(
        pilot,
        lambda: "3 applied" in _painted(host),
        message="receipt never painted '3 applied'",
    )


async def test_the_two_named_skips_also_paint_whole_at_100x30(tmp_path: Path) -> None:
    """AC#6's 100x30 arm: both rows read whole inside their own boxes."""

    vault = (tmp_path / "vault").resolve()
    vault.mkdir()
    write_review_fixture(vault)
    database, interop, scope_service, owner = build_sync_stack(tmp_path)
    await owner.start()
    app = _build_test_app()
    _seed_conversations(app, [])
    app.notes_scope_service = scope_service
    app.notes_sync_runtime_owner = owner
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            controller = await open_keep_synced_review(screen, pilot, host, vault)
            await _wait_for_condition(
                pilot,
                lambda: "Skipped (2)" in _painted(host),
                message=lambda: (
                    "review never painted 'Skipped (2)'; status line: "
                    f"{controller.snapshot.status_line!r}"
                ),
            )
            lines = _painted_review_lines(host, screen)
            assert LATIN1_ROW in lines, lines
            assert MIXED_ROW in lines, lines
            assert "NotesSyncRootRefused" not in _painted(host)
            await _activate_from_review(screen, pilot, host, controller)
            assert database.count_notes() == 3
    finally:
        await owner.shutdown()
        interop.close_all_user_connections()
        database.close_connection()


async def test_check_changes_on_an_active_root_with_a_new_bad_file_names_it_and_never_shows_the_class_name(
    tmp_path: Path,
) -> None:
    """AC#1/#3/#4 on an ACTIVE root, from Manage sync folders.

    After activation a new latin-1 ``legacy.md`` lands and ``ok-3.md`` is
    deleted. The review lists the skip by name beside a real deletion review
    for ``ok-3.md`` (the rest of the folder is still planned), and the root
    row never falls back to ``NotesSyncRootRefused``.
    """

    vault = (tmp_path / "vault").resolve()
    vault.mkdir()
    write_review_fixture(vault)
    database, interop, scope_service, owner = build_sync_stack(tmp_path)
    await owner.start()
    app = _build_test_app()
    _seed_conversations(app, [])
    app.notes_scope_service = scope_service
    app.notes_sync_runtime_owner = owner
    host = LibraryProductionCSSHarness(app)
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            screen = _active_library_screen(host)
            controller = await open_keep_synced_review(screen, pilot, host, vault)
            await _activate_from_review(screen, pilot, host, controller)
            assert database.count_notes() == 3

            (vault / "legacy.md").write_bytes(b"\xe9 legacy\n")
            (vault / "ok-3.md").unlink()

            screen.query_one("#notes-sync-back", Button).press()
            manage = await _wait_for_selector(
                screen, pilot, "#library-notes-manage-sync-folders"
            )
            manage.press()
            check = await _wait_for_selector(screen, pilot, "#notes-sync-root-check-0")
            root_row_before = controller.snapshot.roots[0]
            assert "NotesSyncRootRefused" not in (root_row_before.failure or "")
            check.press()
            await _wait_for_condition(
                pilot,
                lambda: controller.snapshot.phase == "review",
                message=lambda: (
                    "Check changes never reached the review: "
                    f"phase={controller.snapshot.phase!r} "
                    f"status={controller.snapshot.status_line!r} "
                    f"row={controller.snapshot.roots[0].failure!r}"
                ),
            )
            await _wait_for_condition(
                pilot,
                # legacy.md (new) beside latin1.md and mixed.md (still
                # unfixed): every unfixed file is listed again, by name.
                lambda: "Skipped (3)" in _painted(host),
                message=lambda: (
                    "review never painted 'Skipped (3)': "
                    f"{controller.snapshot.status_line!r}"
                ),
            )
            lines = _painted_review_lines(host, screen)
            assert LEGACY_ROW in lines, lines
            assert LATIN1_ROW in lines, lines
            assert MIXED_ROW in lines, lines
            painted = _painted(host)
            assert "ok-3.md" in painted, painted
            assert "Delete/archive" in painted, painted
            assert "NotesSyncRootRefused" not in painted
            assert "NotesSyncRootRefused" not in (
                controller.snapshot.roots[0].failure or ""
            )
            # A Check is mutation-free: nothing was written or deleted.
            assert database.count_notes() == 3
            assert (vault / "legacy.md").read_bytes() == b"\xe9 legacy\n"
            assert not (vault / "ok-3.md").exists()
    finally:
        await owner.shutdown()
        interop.close_all_user_connections()
        database.close_connection()
