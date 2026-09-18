"""Capture-owned reader state through real Local services and production CSS."""

import asyncio

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_library_skill_editor_journeys import _activate
from tldw_chatbook.Library.collections_capture_models import (
    CaptureSaveRequest,
    CollectionsCaptureError,
)


async def _seed():
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    key = scope.active_authority.key
    identities = []
    for title in ("Alpha", "Beta"):
        outcome = await scope.save_capture(
            CaptureSaveRequest(
                key,
                f"https://example.test/{title.lower()}",
                title=title,
                text_content=f"Readable {title} body.",
                freeform_note=f"Saved {title} note",
            )
        )
        identities.append(outcome.capture.identity)
    return app, scope, identities


async def _open(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await screen._select_library_rail_row("browse-collections")
    await _wait_for_selector(screen, pilot, "#library-collections-reader-title")
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._library_collections_capture_controller.state.identity_actions_enabled
        ),
        message="Capture did not settle",
    )
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_annotation_drafts_survive_disclosures_modes_and_capture_return(
    size, theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, scope, identities = await _seed()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    app.notify = host.notify
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        current = screen._library_collections_capture_controller.state.selected_identity
        await _activate(screen, host, pilot, "#library-collections-mode-notes", "Notes")
        await _wait_for_selector(screen, pilot, "#library-collections-freeform-note")
        screen.query_one(
            "#library-collections-freeform-note", TextArea
        ).text = "Unsaved reader note"
        await pilot.pause()
        await _activate(screen, host, pilot, "#library-collections-more", "More")
        await _wait_for_selector(screen, pilot, "#library-collections-summarize")
        assert (
            screen.query_one("#library-collections-freeform-note", TextArea).text
            == "Unsaved reader note"
        )
        await _focus(screen, host, pilot, "#library-collections-more", "More")
        await _activate(screen, host, pilot, "#library-collections-more", "More")
        await _activate(
            screen, host, pilot, "#library-collections-mode-highlights", "Highlights"
        )
        await _wait_for_selector(screen, pilot, "#library-collections-highlight-quote")
        screen.query_one(
            "#library-collections-highlight-quote", TextArea
        ).text = "Unsaved quote"
        screen.query_one(
            "#library-collections-highlight-note", Input
        ).value = "Unsaved highlight note"
        await pilot.pause()
        await _activate(screen, host, pilot, "#library-collections-mode-info", "Info")
        await _focus(screen, host, pilot, "#library-collections-mode-info", "Info")
        other = next(identity for identity in identities if identity != current)
        await screen._select_library_collection_capture(other)
        await pilot.pause()
        await _activate(screen, host, pilot, "#library-collections-mode-notes", "Notes")
        await _wait_for_selector(screen, pilot, "#library-collections-freeform-note")
        assert (
            screen.query_one("#library-collections-freeform-note", TextArea).text
            != "Unsaved reader note"
        )
        await screen._select_library_collection_capture(current)
        await pilot.pause()
        assert (
            screen.query_one("#library-collections-freeform-note", TextArea).text
            == "Unsaved reader note"
        )
        await _activate(
            screen,
            host,
            pilot,
            "#library-collections-freeform-note-save",
            "Save capture note",
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.action_status == "Capture note saved.",
            message="Note save did not settle",
        )
        assert (
            await scope.get_detail(current)
        ).capture.freeform_note == "Unsaved reader note"
        await _focus(
            screen,
            host,
            pilot,
            "#library-collections-freeform-note-save",
            "Save capture note",
        )
        await _activate(
            screen, host, pilot, "#library-collections-mode-highlights", "Highlights"
        )
        await _wait_for_selector(screen, pilot, "#library-collections-highlight-quote")
        assert (
            screen.query_one("#library-collections-highlight-quote", TextArea).text
            == "Unsaved quote"
        )
        assert (
            screen.query_one("#library-collections-highlight-note", Input).value
            == "Unsaved highlight note"
        )
        await _activate(
            screen, host, pilot, "#library-collections-highlight-save", "Add highlight"
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.action_status == "Highlight saved.",
            message="Highlight save did not settle",
        )
        await _wait_for_selector(screen, pilot, "#library-collections-highlight-quote")
        assert (
            screen.query_one("#library-collections-highlight-quote", TextArea).text
            == ""
        )
        assert len((await scope.list_highlights(current)).items) == 1


@pytest.mark.asyncio
async def test_highlights_follow_the_loaded_capture():
    app, scope, identities = await _seed()
    await scope.save_highlight(identities[0], quote="Alpha annotation")
    await scope.save_highlight(identities[1], quote="Beta annotation")
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        await screen._select_library_collection_capture(identities[0])
        await _activate(
            screen, host, pilot, "#library-collections-mode-highlights", "Highlights"
        )
        await _wait_for_condition(
            pilot,
            lambda: bool(screen._collections_state.highlights),
            message="Highlights not loaded",
        )
        await screen._select_library_collection_capture(identities[1])
        await pilot.pause()
        assert [h.quote for h in screen._collections_state.highlights] == [
            "Beta annotation"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_archive_in_saved_scope_loads_successor_and_undo_conflict_is_visible(
    size, theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, scope, _ = await _seed()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        shell = screen.query_one("#library-collections-reader-shell")
        if not shell.effective_layout.library_open:
            await _activate(
                screen, host, pilot, "#library-collections-library-grip", "--->"
            )
        await _activate(
            screen, host, pilot, "#library-collections-scope-saved", "Saved"
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_collections_capture_controller.state.identity_actions_enabled
            ),
            message="Saved scope not ready",
        )
        engine = screen._library_collections_capture_controller
        first = engine.state.selected_identity
        await _activate(
            screen, host, pilot, "#library-collections-archive", "Move to Archive"
        )
        await _wait_for_condition(
            pilot,
            lambda: not engine.state.mutation_loading and not engine.state.page_loading,
            message="Archive not settled",
        )
        assert engine.state.selected_identity != first
        assert (
            engine.state.loaded_detail.capture.identity
            == engine.state.selected_identity
        )
        assert engine.state.identity_actions_enabled
        # Another actor updates the archived capture: its old receipt must fail visibly.
        archived = (await scope.get_detail(first)).capture
        await scope.update_capture(first, archived.revision, {"favorite": True})
        await _activate(
            screen, host, pilot, "#library-collections-archive-undo", "Undo"
        )
        await _wait_for_condition(
            pilot,
            lambda: engine.state.mutation_error is not None,
            message="Expected stale Undo conflict",
        )
        error = await _wait_for_selector(
            screen, pilot, "#library-collections-mutation-error"
        )
        assert "changed" in str(error.renderable).lower()
        assert screen.query_one("#library-collections-reader-retry").focusable
        await _activate(
            screen, host, pilot, "#library-collections-reader-retry", "Refresh reader"
        )
        await _wait_for_condition(
            pilot,
            lambda: engine.state.mutation_error is None,
            message="Refresh did not clear the error",
        )


@pytest.mark.asyncio
async def test_delayed_saved_highlight_refresh_does_not_replace_another_capture(
    monkeypatch,
):
    app, scope, identities = await _seed()
    await scope.save_highlight(identities[1], quote="Beta annotation")
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        await screen._select_library_collection_capture(identities[0])
        await _activate(
            screen, host, pilot, "#library-collections-mode-highlights", "Highlights"
        )
        await _wait_for_selector(screen, pilot, "#library-collections-highlight-quote")
        screen.query_one(
            "#library-collections-highlight-quote", TextArea
        ).text = "Alpha annotation"
        entered, release = asyncio.Event(), asyncio.Event()
        list_highlights = scope.list_highlights

        async def held_list(identity, **kwargs):
            result = await list_highlights(identity, **kwargs)
            if identity == identities[0]:
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(scope, "list_highlights", held_list)
        pending = asyncio.create_task(
            screen.save_library_collection_capture_highlight(
                Button.Pressed(
                    screen.query_one("#library-collections-highlight-save", Button)
                )
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await screen._select_library_collection_capture(identities[1])
            release.set()
            await pending
            await pilot.pause()
            assert [h.quote for h in screen._collections_state.highlights] == [
                "Beta annotation"
            ]
        finally:
            release.set()
            await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_draft", [False, True])
async def test_saved_highlight_survives_failed_refresh_without_resubmitting_draft(
    monkeypatch, newer_draft
):
    app, scope, _ = await _seed()
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        identity = (
            screen._library_collections_capture_controller.state.selected_identity
        )
        await _activate(
            screen, host, pilot, "#library-collections-mode-highlights", "Highlights"
        )
        await _wait_for_selector(screen, pilot, "#library-collections-highlight-quote")
        screen.query_one(
            "#library-collections-highlight-quote", TextArea
        ).text = "Submitted quote"
        entered, release = asyncio.Event(), asyncio.Event()
        save_highlight, list_highlights = scope.save_highlight, scope.list_highlights

        async def held_save(*args, **kwargs):
            entered.set()
            await release.wait()
            return await save_highlight(*args, **kwargs)

        async def failed_list(*args, **kwargs):
            raise CollectionsCaptureError("highlight_refresh_failed")

        monkeypatch.setattr(scope, "save_highlight", held_save)
        monkeypatch.setattr(scope, "list_highlights", failed_list)
        pending = asyncio.create_task(
            screen.save_library_collection_capture_highlight(
                Button.Pressed(
                    screen.query_one("#library-collections-highlight-save", Button)
                )
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if newer_draft:
                screen.query_one(
                    "#library-collections-highlight-quote", TextArea
                ).text = "Newer quote"
            release.set()
            await pending
            await pilot.pause()
            assert [h.quote for h in (await list_highlights(identity)).items] == [
                "Submitted quote"
            ]
            assert "Highlight saved" in screen._collections_state.action_status
            assert "refresh" in screen._collections_state.action_status.lower()
            assert screen.query_one(
                "#library-collections-highlight-quote", TextArea
            ).text == ("Newer quote" if newer_draft else "")
        finally:
            release.set()
            await pending
