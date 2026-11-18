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
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
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
    and after a failed prefetch (task-32103's contradiction). The rail is
    re-synced in place, not recomposed, and Enter on the focused row leaves
    focus on it (review r14ls-rev2: it used to land on a rail control, so a
    second Enter acted on something else).
    """
    from textual.widgets import Static

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Widgets.Library.library_rail import LibraryRail

    app, scope, _identities = await _seed()
    host = LibraryProductionCSSHarness(app)
    composes = []
    compose_content = LibraryScreen.compose_content

    def counting_compose_content(screen):
        composes.append(screen)
        return compose_content(screen)

    rail_composes = []
    rail_compose = LibraryRail.compose

    def counting_rail_compose(rail):
        rail_composes.append(rail)
        return rail_compose(rail)

    def row_label() -> str:
        return str(screen.query_one("#library-row-browse-collections", Button).label)

    monkeypatch.setattr(LibraryScreen, "compose_content", counting_compose_content)
    monkeypatch.setattr(LibraryRail, "compose", counting_rail_compose)
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
        count = 3 if rail_count == "late_capture" else 2
        expected = f"({count})"
        row = screen.query_one("#library-row-browse-collections", Button)
        row.focus()
        await pilot.pause()
        composes.clear()
        rail_composes.clear()
        await pilot.press("enter")

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
        assert len(rail_composes) <= 1, f"{len(rail_composes)} rail composes"
        assert getattr(screen.focused, "id", None) == "library-row-browse-collections"
        title = str(screen.query_one("#library-collections-reader-title").render())
        assert any(name in title for name in ("Alpha", "Beta", "Gamma")), title
        assert len(screen.query(".library-collections-item-row")) == count
        await _wait_for_condition(
            pilot,
            lambda: expected in row_label(),
            message=lambda: f"Rail row reads {row_label()!r}, the canvas {expected}",
        )
        details = str(screen.query_one("#library-details-body", Static).render())
        assert "Collections count unavailable" not in details, details


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
async def test_collections_actions_repaint_their_panes_without_a_library_recompose(
    monkeypatch, tmp_path
):
    """TASK-41126: an action repaints the reader panes, not the Library screen.

    Each of these used to end in a whole-screen recompose (one per action,
    two for a capture). Counts ``compose_content`` from each action until the
    panes show its result; focus lands where that recompose's restore put it:
    the same control, or the content's first one when the control is gone.
    One launch runs them in turn (the UI lane has no time to spare).
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    composes = []
    compose_content = LibraryScreen.compose_content

    def counting_compose_content(screen):
        composes.append(screen)
        return compose_content(screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", counting_compose_content)
    app, _scope, _identities = await _seed()
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        engine = screen._library_collections_capture_controller

        def rows() -> int:
            return len(screen.query(".library-collections-item-row"))

        def work():
            return screen.query_one("#library-collections-work").presentation

        async def act(action, target, done, *, kept_focus=True):
            if target is not None:
                control = await _wait_for_selector(screen, pilot, target)
                control.focus()
                await _wait_for_condition(
                    pilot, lambda: screen.focused is control, message="Never focused"
                )
            focused_id = getattr(screen.focused, "id", None)
            await pilot.pause()
            composes.clear()
            if target is not None:
                await pilot.press("enter")
            else:
                # The FileSave callback's own call (the completion path).
                await screen._export_library_collection_legacy_recovery(
                    tmp_path / "legacy.json"
                )
            await _wait_for_condition(
                pilot,
                lambda: done() and not screen._recompose_required,
                timeout=30,
                message=lambda: (
                    f"{action} never settled: rows={rows()}, "
                    f"status={work().action_status!r}, "
                    f"focus={getattr(screen.focused, 'id', None)}, "
                    f"page_loading={engine.state.page_loading}, "
                    f"confirming={screen._collections_state.confirming_hard_delete}"
                ),
            )
            for _ in range(3):
                await pilot.pause()
            assert not composes, f"{len(composes)} Library recomposes for one {action}"
            expected = (
                screen.query_one(f"#{focused_id}")
                if kept_focus
                else screen._first_focusable_in_content()
            )
            assert screen.focused is expected, (action, screen.focused, expected)

        await _wait_for_condition(
            pilot,
            lambda: rows() == 2
            and not any(
                not worker.is_finished
                and worker.group == "library_collections_capture_entry"
                for worker in screen.app.workers
            ),
            message="Collections entry never settled",
        )
        scope_row = "#library-collections-scope-saved"
        await act(
            "scope change",
            scope_row,
            lambda: work().active_scope == "saved"
            and not engine.state.page_loading
            and not engine.state.detail_loading
            and str(screen.query_one(scope_row, Button).label).startswith("▸"),
        )

        await _activate(
            screen, host, pilot, "#library-collections-mode-highlights", "Highlights"
        )
        quote = await _wait_for_selector(
            screen, pilot, "#library-collections-highlight-quote"
        )
        quote.text = "A quoted line"
        await act(
            "annotate",
            "#library-collections-highlight-save",
            lambda: work().action_status == "Highlight saved."
            and bool(screen.query(".library-collections-highlight-delete")),
        )

        # Delete a seeded capture before Quick Capture adds one: a fresh
        # capture's background extraction bumps its revision, and deleting it
        # meanwhile is refused (revision_conflict), as it should be.
        await _activate(screen, host, pilot, "#library-collections-more", "More")
        await _activate(screen, host, pilot, "#library-collections-hard-delete", "Delete")
        await act(
            "delete",
            "#library-collections-hard-delete-confirm",
            lambda: rows() == 1 and not engine.state.page_loading,
            kept_focus=False,
        )

        await _activate(screen, host, pilot, "#library-collections-quick-capture", "Quick")
        url = await _wait_for_selector(screen, pilot, "#library-collections-capture-url")
        url.value = "https://example.test/gamma"
        await act(
            "capture",
            "#library-collections-capture-save",
            lambda: rows() == 2 and work().action_status.startswith("Saved"),
            kept_focus=False,
        )

        await act(
            "legacy recovery export",
            None,
            lambda: work().action_status.startswith("Legacy recovery export"),
        )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
async def test_unsubmitted_filter_and_link_text_survives_a_pane_repaint():
    """TASK-41126 review r15lc-rev2: typed, unsubmitted text outlives a repaint.

    The filter box, the filter form and the Linked-note ID are read by their
    own actions, not kept in state, so an unrelated action's repaint (More)
    composed them blank. Their own actions still empty them: a link that
    lands clears its ID, and Clear empties the filter fields.
    """
    app, _scope, _identities = await _seed()
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)

        def value(name: str) -> str:
            return screen.query_one(f"#library-collections-{name}", Input).value

        await _wait_for_condition(
            pilot,
            lambda: not any(
                not worker.is_finished
                and worker.group == "library_collections_capture_entry"
                for worker in screen.app.workers
            ),
            message="Collections entry never settled",
        )
        await _activate(screen, host, pilot, "#library-collections-mode-notes", "Notes")
        await _activate(screen, host, pilot, "#library-collections-filters", "Filters")
        typed = {
            "filter": "unsubmitted search",
            "filter-domain": "example.org",
            "filter-tags": "x, y",
            "filter-date-from": "2026-01-01",
            "filter-date-to": "2026-02-01",
            "linked-note-id": "note-42",
        }
        for name, text in typed.items():
            field = await _wait_for_selector(screen, pilot, f"#library-collections-{name}")
            field.value = text
        before = screen.query_one("#library-collections-filter-domain")
        await _activate(screen, host, pilot, "#library-collections-more", "More")
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-collections-hard-delete"))
            and screen.query_one("#library-collections-filter-domain") is not before,
            message="More never repainted the panes",
        )
        for _ in range(3):
            await pilot.pause()
        assert {name: value(name) for name in typed} == typed

        await _activate(
            screen, host, pilot, "#library-collections-linked-note-save", "Link Note"
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.action_status == "Linked Note note-42."
            and value("linked-note-id") == "",
            message=lambda: f"Link left its ID: {value('linked-note-id')!r}",
        )
        await _activate(screen, host, pilot, "#library-collections-filters-clear", "Clear")
        await _wait_for_condition(
            pilot,
            lambda: all(value(name) == "" for name in typed),
            message=lambda: f"Clear left {[value(name) for name in typed]}",
        )


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


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
@pytest.mark.parametrize("entry", ["rail_press", "restored"])
async def test_collections_first_frame_opens_the_library_pane(entry, monkeypatch):
    """TASK-41130: the first Collections compose already has its real layout.

    The shell used to compose with the width-0 layout, which closes and
    disables the Library pane, so the route's focus restore could not land on
    the Collections row; the shell's first layout sync reopened the pane and
    a one-time retry (r14ls-rev2/rev3) put focus back. The layout sync is held
    back here, so what is checked is the first frame itself: the pane is open
    and enabled, and Enter on the row leaves focus on the row.
    """
    from tldw_chatbook.UI.Library_Modules.library_collections_controller import (
        LibraryCollectionsController,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    hold = {"on": entry == "restored"}
    layout_sync = (
        LibraryCollectionsController._sync_library_collections_reader_layout_from_shell
    )

    def held_layout_sync(controller, priority=None):
        if not hold["on"]:
            layout_sync(controller, priority)

    monkeypatch.setattr(
        LibraryCollectionsController,
        "_sync_library_collections_reader_layout_from_shell",
        held_layout_sync,
    )
    app, _scope, _identities = await _seed()
    screen = LibraryScreen(app)
    if entry == "restored":
        screen._library_selected_row_id = "browse-collections"
    host = LibraryProductionCSSHarness(app, screen=screen)
    async with host.run_test(size=(170, 48)) as pilot:
        await _wait_for_library_shell(screen, pilot)

        def row() -> Button:
            return screen.query_one("#library-row-browse-collections", Button)

        if entry == "rail_press":
            await _wait_for_condition(
                pilot, lambda: "(2)" in str(row().label), message="Prefetch never painted"
            )
            row().focus()
            await pilot.pause()
            hold["on"] = True
            await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-collections-reader-shell"))
            and not screen._recompose_required
            and not any(
                worker.group == "library_collections_capture_entry"
                and not worker.is_finished
                for worker in screen.app.workers
            ),
            message="Collections never settled",
        )
        for _ in range(3):
            await pilot.pause()
        shell = screen.query_one("#library-collections-reader-shell")
        assert shell.effective_layout.library_open, shell.effective_layout
        assert shell.library.display and not shell.library.disabled
        if entry == "rail_press":
            assert screen.focused is row(), screen.focused
            hold["on"] = False
            screen._sync_library_collections_reader_layout_from_shell()
            for _ in range(3):
                await pilot.pause()
            assert screen.focused is row(), screen.focused


async def _settle_collections_entry(screen, pilot):
    await _wait_for_condition(
        pilot,
        lambda: (
            bool(screen.query("#library-collections-reader-shell"))
            and not screen._recompose_required
            and not any(
                worker.group == "library_collections_capture_entry"
                and not worker.is_finished
                for worker in screen.app.workers
            )
        ),
        message="Collections never settled",
    )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
@pytest.mark.parametrize("terminal_width", [80, 127])
async def test_a_pane_toggle_sent_before_the_shell_is_measured_lands_once_it_is(
    terminal_width, monkeypatch
):
    """TASK-41503: a toggle that arrives before the first measure is kept.

    At 80 columns Collections opens on its reader stage, Library closed. A
    grip press (or Escape) there reopens Library by giving it priority, and
    the preference it writes is already "open". While the shell's allocation
    is not yet current the layout sync returns early, so that priority was
    dropped and the measured sync resolved the stage again: the first press
    did nothing.

    127 (review r16mo-rev1): the terminal widens before the measure, so the
    stored layout still has Library closed from 80 columns, and the shell is
    then 123 cells wide. Kept only as the stored layout's inherited priority,
    the toggle lost to the resolver's reopen hysteresis there (a 122-125 cell
    shell with Items open); it has to reach the measured resolve as an
    explicit priority.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
        AdaptiveReaderShellResized,
    )

    measured = {"on": False}
    allocation_is_current = LibraryScreen._library_adaptive_reader_allocation_is_current

    def unmeasured_until_released(screen, reader):
        return measured["on"] and allocation_is_current(screen, reader)

    monkeypatch.setattr(
        LibraryScreen,
        "_library_adaptive_reader_allocation_is_current",
        unmeasured_until_released,
    )
    app, _scope, _identities = await _seed()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = "browse-collections"
    host = LibraryProductionCSSHarness(app, screen=screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await _settle_collections_entry(screen, pilot)
        if terminal_width != 80:
            await pilot.resize_terminal(terminal_width, 40)
            await _settle_collections_entry(screen, pilot)
        shell = screen.query_one("#library-collections-reader-shell")
        assert not shell.effective_layout.library_open, shell.effective_layout
        if terminal_width != 80:
            await _wait_for_condition(
                pilot,
                lambda: 122 <= shell.content_size.width <= 125,
                message=lambda: f"Shell left the band: {shell.content_size.width}",
            )

        claimed = screen._library_reader_persistence_generations["library"]
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_reader_persistence_generations["library"] > claimed,
            message="The grip press never reached the toggle",
        )
        assert not shell.effective_layout.library_open, "applied before any measure"

        measured["on"] = True
        shell.post_message(AdaptiveReaderShellResized())
        await _wait_for_condition(
            pilot,
            lambda: shell.effective_layout.library_open,
            message=lambda: f"The early toggle was dropped: {shell.effective_layout}",
        )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile  # the Library harness's profile (TASK-39141)
async def test_a_hard_delete_refused_as_stale_reloads_so_the_next_confirm_deletes():
    """TASK-41503: a ``revision_conflict`` refusal reloads the capture.

    The confirm sends the loaded capture's revision. When another writer
    has moved the capture on, the delete is refused; the reader used to keep
    the stale revision, so every later confirm was refused too until the
    capture was picked again.
    """
    app, scope, _identities = await _seed()
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        engine = screen._library_collections_capture_controller
        await _settle_collections_entry(screen, pilot)

        def rows() -> int:
            return len(screen.query(".library-collections-item-row"))

        await _activate(screen, host, pilot, "#library-collections-more", "More")
        await _activate(
            screen, host, pilot, "#library-collections-hard-delete", "Delete"
        )
        await _wait_for_selector(
            screen, pilot, "#library-collections-hard-delete-confirm"
        )
        stale = engine.state.loaded_detail.capture
        # Another writer moves the open capture on behind the reader.
        await scope.update_capture(stale.identity, stale.revision, {"favorite": True})
        warnings = []
        app.notify = lambda message, **_kwargs: warnings.append(message)

        await _activate(
            screen, host, pilot, "#library-collections-hard-delete-confirm", "Delete"
        )
        await _wait_for_condition(
            pilot,
            lambda: "revision_conflict" in warnings,
            message=lambda: f"The stale delete was not refused: {warnings}",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                engine.state.loaded_detail.capture.revision > stale.revision
                and not engine.state.detail_loading
            ),
            message="The refused delete never reloaded the capture",
        )
        assert rows() == 2

        await _activate(
            screen, host, pilot, "#library-collections-hard-delete-confirm", "Delete"
        )
        await _wait_for_condition(
            pilot,
            lambda: rows() == 1 and not engine.state.page_loading,
            message=lambda: f"The next confirm did not delete: {warnings}",
        )
