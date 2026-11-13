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


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
@pytest.mark.parametrize("rail_count", ["late_capture", "failed_prefetch"])
async def test_rail_press_recomposes_library_once_and_loads_captures_in_place(
    rail_count, monkeypatch
):
    """TASK-40328: the capture loads re-render the reader panes, not the screen.

    The entry loader used to end each of its three steps in a whole-screen
    recompose (rail, canvas and reader re-mounted three times, about
    0.64-0.80 s of UI-loop CPU), on top of the route change's own one.
    Counts ``compose_content`` passes instead of timing the press.

    Those recomposes were also what repainted the rail with the count the
    load produced (review r14ls-rev1): the painted Collections row and the
    Details line must show it, for a capture saved after the rail's prefetch
    and after a failed prefetch (task-32103's contradiction).
    """
    from textual.widgets import Static

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app, scope, _identities = await _seed()
    host = LibraryProductionCSSHarness(app)
    composes = []
    compose_content = LibraryScreen.compose_content

    def counting_compose_content(screen):
        composes.append(screen)
        return compose_content(screen)

    def row_label() -> str:
        return str(screen.query_one("#library-row-browse-collections", Button).label)

    monkeypatch.setattr(LibraryScreen, "compose_content", counting_compose_content)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_condition(
            pilot, lambda: "(2)" in row_label(), message="Prefetch never painted"
        )
        if rail_count == "late_capture":
            await scope.save_capture(
                CaptureSaveRequest(
                    scope.active_authority.key,
                    "https://example.test/gamma",
                    title="Gamma",
                    text_content="Readable Gamma body.",
                )
            )
        else:
            screen._library_collections_prefetched_total = None
            screen._library_collections_count_authority = scope.active_authority.key
            screen._library_collections_count_failure = "error"
            await screen.recompose()
            await _wait_for_condition(
                pilot, lambda: "(\u2014)" in row_label(), message="No failed count"
            )
            assert "Collections count unavailable" in str(
                screen.query_one("#library-details-body", Static).render()
            )
        expected = "(3)" if rail_count == "late_capture" else "(2)"
        composes.clear()
        screen.query_one("#library-row-browse-collections", Button).press()

        def settled() -> bool:
            work = screen.query("#library-collections-work")
            return (
                bool(work)
                and work.first().presentation.state.loaded_detail is not None
                and not screen._recompose_required
                and not any(
                    worker.group == "library_collections_capture_entry"
                    and not worker.is_finished
                    for worker in screen.app.workers
                )
            )

        await _wait_for_condition(pilot, settled, message="Collections never settled")
        await pilot.pause()
        await pilot.pause()
        assert len(composes) <= 1, f"{len(composes)} Library recomposes for one press"
        title = str(screen.query_one("#library-collections-reader-title").render())
        assert any(name in title for name in ("Alpha", "Beta", "Gamma")), title
        assert len(screen.query(".library-collections-item-row")) == int(expected[1])
        await _wait_for_condition(
            pilot,
            lambda: expected in row_label(),
            message=lambda: f"Rail row reads {row_label()!r}, the canvas {expected}",
        )
        details = str(screen.query_one("#library-details-body", Static).render())
        assert "Collections count unavailable" not in details, details


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
async def test_a_pane_repaint_cancelled_mid_recompose_is_rebuilt_by_the_next():
    """TASK-40328 review r14ls-rev1: a cancelled repaint never strands a pane.

    The exclusive entry worker is cancelled when the rail is pressed again,
    possibly while a pane is mid-recompose: its old children are already
    being removed and the new ones never mount. The next repaint must rebuild
    that pane even when it carries the very same presentation.
    """
    import dataclasses

    from tldw_chatbook.Widgets.Library.library_collections_capture_reader import (
        recompose_capture_panes,
    )

    app, _scope, _identities = await _seed()
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        items = screen.query_one("#library-collections-items")

        def rows() -> int:
            return len(items.query(".library-collections-item-row"))

        assert rows() == 2
        presentation = dataclasses.replace(
            screen._collections_controller._library_collections_capture_presentation(),
            action_status="restarted",
        )
        original_recompose = items.recompose

        async def cancelled_mid_recompose() -> None:
            asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
            await original_recompose()

        items.recompose = cancelled_mid_recompose
        repaint = asyncio.create_task(recompose_capture_panes(screen, presentation))
        with pytest.raises(asyncio.CancelledError):
            await repaint
        del items.recompose
        await _wait_for_condition(
            pilot, lambda: rows() == 0, message="The cancel left the old rows"
        )

        await recompose_capture_panes(screen, presentation)
        await _wait_for_condition(
            pilot,
            lambda: rows() == 2,
            message=lambda: f"An identical repaint left the pane with {rows()} rows",
        )
