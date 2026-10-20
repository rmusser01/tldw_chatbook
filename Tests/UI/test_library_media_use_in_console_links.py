"""TASK-34000.24: Media "Use in Console" links the item on use and names the
workspace, never its raw id.

The Media reader's "Use in Console" (and `c`) used to refuse every item on a
fresh profile with "Copy or link this media into workspace workspace-default
before using it in Console." -- an id the user has never seen and an action
that does not exist for media -- while Search ▸ evidence ▸ `u` staged the same
item with no workspace rule at all. Notes and conversations already link on
use (task-32536, task-32107). These Pilot arms pin the media port: one press
links AND stages, the receipt names "Local Default", the link that press
added can be undone, `c` and `u` agree, a block a link cannot resolve still
refuses in plain language, no active workspace stays the global scope
(task-15120), no hand-off copy for any item type carries a raw workspace id,
a stale receipt is withheld after a workspace switch, and (fix round 1) an
Undo pressed after the workspace moved or the membership went away says so
and clears the dead control instead of looping, and Search `u` unlinks its
own write when Console refuses the stage.

Fresh-profile shape on purpose: the seeded media item is NOT linked to the
active workspace (the critique's exact state), so the gate is closed when the
button is pressed. The seeded record carries a production-shaped id (``"1"``,
what ``LocalMediaReadingService`` returns), so the reader's payload id, the
row model's id and the Search result's parsed record id are the same string
-- the ``media-N`` fixture spelling would make `c` and `u` disagree only in
the fixture. Real ``LocalWorkspaceRegistryService`` (the test app's own), real
app stylesheet (``LibraryHarness``), painted text through the compositor,
membership facts read back from the registry.

Three boots: the reader arms are one journey on one app (the
``test_nav_veto_keeps_highlight`` precedent for the 35 s lane budget), each
arm a labelled section with the same assertions it would carry alone; Search
``u`` needs its own fake retrieval service, so it boots twice more. The
43-character workspace-name arm, the 160x45 fit and the off-snapshot item
(I-1) are in ``test_library_media_use_in_console_links_extended.py``, outside
the lane.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _StaticLibraryRagSearchService,
    _active_library_screen,
    _build_test_app,
    _painted_rows,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Workspaces import DEFAULT_WORKSPACE_ID
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

LINKED_RECEIPT = "Linked to Local Default · staged in Console"
ALREADY_LINKED_RECEIPT = "Already linked to Local Default · staged in Console"
FAILED_LINK_LINE = (
    "Can't use this media in Console — it could not be linked to Local Default. "
    "Next: try again."
)
RAW_ID = DEFAULT_WORKSPACE_ID
THESIS_ID = "ws-thesis-9f2c"
MEDIA_ID = "1"


def _production_shaped_media():
    """The two-item fixture's first record with a real local id."""
    return [{**_two_media_items()[0], "id": MEDIA_ID}]


def _media_host(*, notes=None, conversations=None, media=None):
    app = _build_test_app()
    _seed_conversations(
        app,
        conversations or [],
        notes=notes,
        media=media if media is not None else _production_shaped_media(),
    )
    app.open_chat_with_handoff = Mock()
    app.open_console_for_live_work = Mock()
    app.notify = Mock()
    return LibraryHarness(app), app


def _memberships(app, item_type="media", item_id=MEDIA_ID):
    return app.workspace_registry_service.get_item_memberships(
        item_type=item_type, item_id=item_id
    )


def _notified(app) -> list[str]:
    return [str(call.args[0]) for call in app.notify.call_args_list if call.args]


def _painted(screen) -> str:
    return "\n".join(_painted_rows(screen))


async def _open_media_reader(screen, pilot) -> None:
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    screen.query_one("#library-media-row-0").press()
    await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")
    await _wait_for_condition(
        pilot,
        lambda: screen._media_state.reader_session.loaded_id is not None
        and screen._media_state.reader_session.pending_request is None,
        message="media reader never settled",
    )


async def _press_use_in_console(screen, pilot) -> None:
    screen.query_one("#library-media-use-in-chat", Button).press()
    await pilot.pause()
    await pilot.pause()


async def _press_undo(screen, pilot) -> None:
    screen.query_one("#library-media-link-undo", Button).press()
    await pilot.pause()
    await pilot.pause()


def _receipt_widget_text(screen) -> str:
    receipts = list(screen.query("#library-media-link-receipt"))
    return str(receipts[0].renderable) if receipts else ""


async def _wait_for_receipt(screen, pilot, text: str, app=None) -> None:
    """The widget carries the whole line, and the whole default "Linked to
    Local Default · staged in Console" line is painted beside "Undo link"
    (fix round 1, M-9). The "Already linked to …" variant is 51 characters:
    with a 9-cell Undo on the same row it cannot fit a 56-cell Reader pane
    (160x45 and 170x48 both give it 56), so for that spelling the painted
    check is the head through the workspace name, and the "Undo link"
    control must still be painted. A long custom name ellipsises; the
    extended file pins that."""
    painted_head = text if text.startswith("Linked to") else text.split(" · ")[0]
    await _wait_for_condition(
        pilot,
        lambda: _receipt_widget_text(screen) == text
        and painted_head in _painted(screen),
        message=lambda: (
            f"receipt {text!r} never painted; widget={_receipt_widget_text(screen)!r}; "
            f"painted rows with 'inked'={[r for r in _painted_rows(screen) if 'inked' in r]!r}; "
            f"notified={_notified(app) if app is not None else '?'!r}"
        ),
    )


async def _wait_for_no_receipt(screen, pilot, why: str) -> None:
    """Fix round 1, M-2: the widget is gone AND neither spelling is painted
    (an "Already linked to …" receipt is not caught by a "Linked to" check)."""

    def gone() -> bool:
        painted = _painted(screen)
        return (
            not screen.query("#library-media-link-receipt")
            and not screen.query("#library-media-link-undo")
            and "Linked to" not in painted
            and "linked to" not in painted
        )

    await _wait_for_condition(pilot, gone, message=why)


def _assert_rows_name_the_workspace(screen, name: str, raw_id: str) -> None:
    """AC#3, AC#5: every blocked row's copy names the workspace, never its id."""
    state = screen._library_workspace_depth_state(refresh=True)
    by_type = {row.item_type: row for row in state.source_rows}
    assert set(by_type) >= {"media", "note", "conversation"}, by_type
    for item_type, row in by_type.items():
        if row.active_context_eligible:
            continue
        assert row.recovery_copy == (
            f"Copy or link this {item_type} into the {name} workspace before "
            "using it in Console."
        ), (item_type, row.recovery_copy)
        assert raw_id not in row.recovery_copy
    assert raw_id not in state.context_handoff_tooltip


async def test_media_use_in_console_links_on_use_journey():
    """One app, one open reader, the whole hand-off story in order.

    Sections: AC#3/#5 copy on a fresh profile -> AC#1/#2/#5 first press links
    and stages with the receipt -> second press says "Already linked" ->
    AC#2/#6 Undo removes the membership -> M-4 an Undo after the membership
    went away elsewhere clears the receipt without a retry loop -> AC#4 a
    failed link refuses in plain language, nothing staged -> AC#1 the `c`
    key takes the same path -> M-3 an Undo pressed after the workspace moved
    says so and drops the dead control -> Review Focus 2 the next press
    links into the new workspace, named -> AC#4 no active workspace is the
    global scope (task-15120): stages, links nothing.
    """
    host, app = _media_host(
        notes=_two_notes()[:1], conversations=_two_conversations()[:1]
    )
    registry = app.workspace_registry_service
    active = registry.get_active_workspace()
    assert active is not None and active.workspace_id == RAW_ID
    real_link = registry.link_membership

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # -- AC#3, AC#5: fresh profile, every row blocked, copy names the
        # workspace for media, notes and conversations alike.
        _assert_rows_name_the_workspace(screen, "Local Default", RAW_ID)

        await _open_media_reader(screen, pilot)
        assert _memberships(app) == (), "control: the fresh profile has no link"
        assert "Linked to" not in _painted(screen)

        # -- AC#1, AC#2, AC#5: one press links AND stages; the receipt names
        # the workspace by display name; the membership uses the SAME id the
        # row model reads for this record.
        await _press_use_in_console(screen, pilot)
        rows = _memberships(app)
        assert [(row.workspace_id, row.item_id) for row in rows] == [(RAW_ID, MEDIA_ID)]
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.item_type == "media" and payload.source_id == MEDIA_ID
        await _wait_for_receipt(screen, pilot, LINKED_RECEIPT, app)
        # Painted, not merely mounted: `#library-media-viewer Static {width:
        # 100%}` once gave the receipt the whole row and pushed the Undo off
        # the pane while its `.display` read True (found live, 160x45).
        undo = screen.query_one("#library-media-link-undo", Button)
        receipt = screen.query_one("#library-media-link-receipt")
        assert "Undo link" in _painted(screen), _painted(screen)
        assert undo.region.width >= len("Undo link") and undo.region.height == 1
        assert undo.region.y == receipt.region.y, (undo.region, receipt.region)
        assert undo.region.right <= screen.size.width
        assert not [line for line in _notified(app) if RAW_ID in line], _notified(app)
        assert RAW_ID not in _painted(screen)

        # -- second press: nothing more is written, and it says so.
        await _press_use_in_console(screen, pilot)
        await _wait_for_receipt(screen, pilot, ALREADY_LINKED_RECEIPT, app)
        assert len(_memberships(app)) == 1
        assert app.open_chat_with_handoff.call_count == 2
        # The earlier press of this session wrote it, so Undo stays offered.
        assert "Undo link" in _painted(screen)

        # -- AC#2, AC#6: Undo removes the membership that press added.
        await _press_undo(screen, pilot)
        assert _memberships(app) == ()
        await _wait_for_no_receipt(screen, pilot, "receipt still painted after Undo")
        gate = screen._library_workspace_depth_state(refresh=True)
        row = next(r for r in gate.source_rows if r.item_id == MEDIA_ID)
        assert row.active_context_eligible is False
        assert "Local Default" in row.recovery_copy and RAW_ID not in row.recovery_copy

        # -- fix round 1, M-4: the membership went away elsewhere (another
        # surface, another device); Undo is the outcome the user wanted, so
        # the receipt clears without a "could not be undone" loop.
        await _press_use_in_console(screen, pilot)
        await _wait_for_receipt(screen, pilot, LINKED_RECEIPT, app)
        assert registry.unlink_membership(RAW_ID, item_type="media", item_id=MEDIA_ID)
        assert _memberships(app) == ()
        await _press_undo(screen, pilot)
        await _wait_for_no_receipt(
            screen, pilot, "receipt still painted after an Undo of a gone membership"
        )
        assert not [line for line in _notified(app) if "could not be undone" in line], (
            _notified(app)
        )

        # -- AC#4: a block a link cannot resolve (the registry refuses the
        # write) still refuses, in plain language, naming the workspace by
        # name with a next step; nothing staged, nothing linked.
        def failing_link(*args, **kwargs):
            raise WorkspaceRegistryServiceError("storage failure")

        registry.link_membership = failing_link
        staged_before = app.open_chat_with_handoff.call_count
        await _press_use_in_console(screen, pilot)
        assert _memberships(app) == ()
        assert app.open_chat_with_handoff.call_count == staged_before
        assert FAILED_LINK_LINE in _notified(app), _notified(app)
        assert not [line for line in _notified(app) if RAW_ID in line]
        assert "Linked to" not in _painted(screen)
        registry.link_membership = real_link

        # -- AC#1: the `c` key takes the same path as the button.
        screen.query_one("#library-media-use-in-chat", Button).focus()
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        await pilot.pause()
        assert [(row.workspace_id, row.item_id) for row in _memberships(app)] == [
            (RAW_ID, MEDIA_ID)
        ]
        assert app.open_chat_with_handoff.call_count == staged_before + 1
        await _wait_for_receipt(screen, pilot, LINKED_RECEIPT, app)

        # -- fix round 1, M-3: the workspace moved under a standing receipt
        # (switched from Console, say) and nothing re-synced the Reader yet.
        # The Undo that is still on screen must not remove a membership of a
        # workspace that is no longer active: it says why, once, and the
        # dead control disappears.
        thesis = registry.create_workspace(workspace_id=THESIS_ID, name="Thesis")
        registry.set_active_workspace(thesis.workspace_id)
        assert screen.query("#library-media-link-undo"), "control: Undo still mounted"
        await _press_undo(screen, pilot)
        assert [row.workspace_id for row in _memberships(app)] == [RAW_ID], (
            "the Local Default membership must survive a stale Undo"
        )
        stale_lines = [line for line in _notified(app) if "no longer active" in line]
        assert len(stale_lines) == 1, _notified(app)
        assert THESIS_ID not in stale_lines[0] and RAW_ID not in stale_lines[0]
        await _wait_for_no_receipt(
            screen, pilot, "stale receipt still painted after the workspace switch"
        )

        # -- Review Focus 2: the next press links into the NEW active
        # workspace, by name.
        screen._invalidate_library_workspace_depth_state()
        _assert_rows_name_the_workspace(screen, "Thesis", THESIS_ID)
        await _press_use_in_console(screen, pilot)
        await _wait_for_receipt(screen, pilot, "Linked to Thesis · staged in Console", app)
        assert sorted(row.workspace_id for row in _memberships(app)) == sorted(
            [RAW_ID, THESIS_ID]
        )
        assert THESIS_ID not in _painted(screen)
        assert not [line for line in _notified(app) if THESIS_ID in line]

        # -- AC#4 last clause / task-15120: no active workspace IS the global
        # scope -- the item stages as before, links nothing, shows no
        # receipt, and no copy carries a raw id.
        registry.clear_active_workspace()
        screen._invalidate_library_workspace_depth_state()
        screen._sync_library_media_viewer_or_recompose()
        await pilot.pause()
        await _wait_for_no_receipt(
            screen, pilot, "receipt painted with no active workspace"
        )
        memberships_before = len(_memberships(app))
        staged_before = app.open_chat_with_handoff.call_count
        await _press_use_in_console(screen, pilot)
        assert len(_memberships(app)) == memberships_before
        assert app.open_chat_with_handoff.call_count == staged_before + 1
        assert "Linked to" not in _painted(screen)
        assert not [
            line for line in _notified(app) if RAW_ID in line or THESIS_ID in line
        ]


# --- AC#4: Search `u` and the reader's `c` give the same result


def _search_result_for(item_id: str) -> dict:
    return {
        "title": "Interview Recording",
        "snippet": "Matched media · audio",
        "source_id": item_id,
        "provenance": {"source_type": "media"},
    }


async def _run_search_and_select_first(screen, pilot) -> None:
    screen.query_one("#library-row-browse-search").press()
    await _wait_for_selector(screen, pilot, "#library-rag-query-input")
    screen.query_one("#library-rag-query-input", Input).value = "interview"
    await _wait_for_library_rag_query_ready(screen, pilot, "interview")
    screen.query_one("#library-rag-run-query", Button).press()
    await _wait_for_selector(screen, pilot, "#library-rag-select-result-0")
    screen.query_one("#library-rag-select-result-0", Button).press()
    await pilot.pause()


async def test_search_u_and_reader_c_agree():
    host, app = _media_host()
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        {"results": [_search_result_for(MEDIA_ID)]}
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_search_and_select_first(screen, pilot)
        assert _memberships(app) == (), "control: nothing linked before u"

        await pilot.press("u")
        await pilot.pause()
        await pilot.pause()

        app.open_console_for_live_work.assert_called_once()
        assert app.open_console_for_live_work.call_args.kwargs["title"] == (
            "Interview Recording"
        )
        assert [(row.workspace_id, row.item_id) for row in _memberships(app)] == [
            (RAW_ID, MEDIA_ID)
        ]
        assert LINKED_RECEIPT in _notified(app), _notified(app)
        assert not [line for line in _notified(app) if RAW_ID in line]

        # The reader now finds the item already linked: same membership, no
        # second write, the "Already linked" receipt.
        await _open_media_reader(screen, pilot)
        await _press_use_in_console(screen, pilot)
        await _wait_for_receipt(screen, pilot, ALREADY_LINKED_RECEIPT, app)
        assert len(_memberships(app)) == 1
        app.open_chat_with_handoff.assert_called_once()
        # This press wrote nothing, and no earlier press here did: no Undo.
        assert not screen.query("#library-media-link-undo")


async def test_search_u_unlinks_its_own_write_when_console_refuses():
    """Fix round 1, I-2: the receipt is said only AFTER Console took the
    stage, and a refused stage takes the membership `u` just wrote back with
    it (the reader's own rule) -- no "Linked to …" beside "could not be
    staged", no orphan membership."""
    host, app = _media_host()
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        {"results": [_search_result_for(MEDIA_ID)]}
    )
    app.open_console_for_live_work = Mock(return_value=False)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _run_search_and_select_first(screen, pilot)

        await pilot.press("u")
        await pilot.pause()
        await pilot.pause()

        app.open_console_for_live_work.assert_called_once()
        assert _memberships(app) == (), "the refused stage's own write was not undone"
        assert not [line for line in _notified(app) if "Linked to" in line], (
            _notified(app)
        )
