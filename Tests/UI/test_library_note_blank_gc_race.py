"""Real-database controls for blank-note GC crossing an autosave reply."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import TextArea

from Tests.console_resource_fixtures import (  # noqa: F401
    close_owned_console_resources,
    close_owned_console_test_apps,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _press_note_back,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
    real_notes_scope_service as _real_notes_scope_service,
)
from tldw_chatbook.Library.library_notes_session import PortSaveKind
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens import library_screen as library_screen_module

real_notes_scope_service = _real_notes_scope_service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intervening_change", [None, "external", "external_at_delete", "authored"]
)
async def test_blank_note_back_gcs_after_committed_autosave_reply(
    real_notes_scope_service: NotesScopeService,
    monkeypatch: pytest.MonkeyPatch,
    intervening_change: str | None,
) -> None:
    """Back must settle its own successful save before version-checked GC.

    Args:
        real_notes_scope_service: Exact-owned, real SQLite Notes service.
        monkeypatch: Scoped save-response and navigation scheduling controls.
        intervening_change: An external write or new canonical draft during Back.
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = real_notes_scope_service
    unrelated = await service.save_note(
        scope="local_note",
        title="Unrelated note",
        content="Keep this note",
        user_id="default_user",
        keywords=[],
    )
    app.notes_scope_service = service
    host = LibraryHarness(app)
    committed = asyncio.Event()
    release_reply = asyncio.Event()
    deletes = []
    ordering = {}
    release_task = None

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        screen.query_one("#library-notes-create-blank").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.editor_armed,
            message="The created note editor did not arm.",
        )
        session = screen._library_note_session
        note_id = screen._notes_state.selected_note_id
        original_save = session._port.save_note
        original_delete = service.delete_note
        original_flush = screen._flush_library_note_save

        async def hold_committed_reply(*args, **kwargs):
            reply = await original_save(*args, **kwargs)
            if not committed.is_set():
                assert reply.kind is PortSaveKind.SAVED
                committed.set()
                await release_reply.wait()
            return reply

        async def record_real_delete(**kwargs):
            deletes.append(kwargs.copy())
            if intervening_change == "external_at_delete":
                before_delete = await service.get_note_detail(
                    scope="local_note", note_id=note_id, user_id="default_user"
                )
                assert kwargs["version"] == before_delete["version"]
                await service.save_note(
                    scope="local_note",
                    note_id=note_id,
                    version=before_delete["version"],
                    title=before_delete["title"],
                    content="external edit",
                    keywords=[],
                    user_id="default_user",
                )
            try:
                result = await original_delete(**kwargs)
            except Exception as error:
                deletes[-1]["error_type"] = type(error).__name__
                raise
            deletes[-1]["result"] = result
            return result

        async def change_then_release():
            try:
                if intervening_change == "external":
                    await service.save_note(
                        scope="local_note",
                        note_id=note_id,
                        version=detail["version"],
                        title=detail["title"],
                        content="external edit",
                        keywords=[],
                        user_id="default_user",
                    )
                elif intervening_change == "authored":
                    assert session.mutate(body="new authored draft")
            finally:
                release_reply.set()

        def dispatch_release():
            nonlocal release_task
            release_task = asyncio.create_task(change_then_release())

        async def release_at_flush_yield():
            # Existing GC captures its version before this first yield. A
            # pending-save barrier instead accepts the real reply first.
            asyncio.get_running_loop().call_soon(dispatch_release)
            return await original_flush()

        monkeypatch.setattr(session._port, "save_note", hold_committed_reply)
        monkeypatch.setattr(service, "delete_note", record_real_delete)
        monkeypatch.setattr(screen, "_flush_library_note_save", release_at_flush_yield)
        try:
            screen.query_one("#library-note-body", TextArea).focus()
            screen.query_one("#library-note-body", TextArea).text = "autosaved draft"
            await asyncio.wait_for(committed.wait(), timeout=5)
            detail = await service.get_note_detail(
                scope="local_note", note_id=note_id, user_id="default_user"
            )
            assert detail["content"] == "autosaved draft"
            assert detail["version"] == session.snapshot.version + 1
            assert session.snapshot.saving
            assert session.destructive_admission is None
            assert not session.destructive_running
            assert screen._notes_state.session_blank_id == note_id
            ordering.update(
                committed_version=detail["version"],
                session_version=session.snapshot.version,
                saving=session.snapshot.saving,
                destructive_admission=session.destructive_admission,
                destructive_running=session.destructive_running,
            )

            screen.query_one("#library-note-body", TextArea).text = ""
            await _wait_for_condition(
                pilot,
                lambda: session.snapshot.body == "",
                message="The empty body did not reach the canonical draft.",
            )
            assert session.snapshot.saving
            _press_note_back(screen)
            await _wait_for_condition(
                pilot,
                lambda: not screen._notes_state.selected_note_id,
                message=lambda: (
                    "Back did not leave the blank note editor: "
                    f"selected={screen._notes_state.selected_note_id!r}, "
                    f"released={release_reply.is_set()}, deletes={deletes!r}, "
                    f"saving={session.snapshot.saving if session.snapshot else None}"
                ),
            )
            if release_task is not None:
                await release_task
            if session._save_task is not None:
                await asyncio.wait_for(asyncio.shield(session._save_task), timeout=5)
            count = await service.count_notes(
                scope="local_note", user_id="default_user"
            )
            evidence = f"ordering={ordering!r}; deletes={deletes!r}"
            unrelated_detail = await service.get_note_detail(
                scope="local_note", note_id=unrelated["id"], user_id="default_user"
            )
            assert unrelated_detail["content"] == "Keep this note"
            assert unrelated_detail["version"] == unrelated["version"]
            assert all(deletion["note_id"] == note_id for deletion in deletes)
            if intervening_change is None:
                assert count == 1, f"Blank note survived Back: {evidence}"
                assert len(deletes) == 1 and deletes[0]["result"] is True, evidence
            else:
                assert count == 2, (
                    f"Intervening content was silently deleted: {evidence}"
                )
                final_detail = await service.get_note_detail(
                    scope="local_note", note_id=note_id, user_id="default_user"
                )
                expected = (
                    "external edit"
                    if intervening_change.startswith("external")
                    else "new authored draft"
                )
                assert final_detail["content"] == expected, evidence
                if intervening_change == "external_at_delete":
                    assert len(deletes) == 1, evidence
                    assert deletes[0]["error_type"] == "ConflictError", evidence
                    assert deletes[0]["version"] + 1 == final_detail["version"], (
                        evidence
                    )
                else:
                    assert deletes == [], evidence
        finally:
            release_reply.set()
            if release_task is not None:
                await release_task
            if session._save_task is not None:
                await asyncio.wait_for(asyncio.shield(session._save_task), timeout=5)
