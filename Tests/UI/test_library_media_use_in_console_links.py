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
and a stale receipt is withheld after a workspace switch.

Fresh-profile shape on purpose: the seeded media item is NOT linked to the
active workspace (the critique's exact state), so the gate is closed when the
button is pressed. Real ``LocalWorkspaceRegistryService`` (the test app's
own), real app stylesheet (``LibraryHarness``), painted text through the
compositor, membership facts read back from the registry.

Two boots, not eight: the reader arms are one journey on one app (the
``test_nav_veto_keeps_highlight`` precedent for the 35 s lane budget), each
arm a labelled section with the same assertions it would carry alone; Search
``u`` needs its own fake retrieval service, so it boots once more. The
40-character workspace-name arm (Review Focus 5) is in
``test_library_media_use_in_console_links_extended.py``, outside the lane.
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


def _media_host(*, notes=None, conversations=None):
    app = _build_test_app()
    _seed_conversations(
        app,
        conversations or [],
        notes=notes,
        media=_two_media_items()[:1],
    )
    app.open_chat_with_handoff = Mock()
    app.open_console_for_live_work = Mock()
    app.notify = Mock()
    return LibraryHarness(app), app


def _memberships(app, item_type="media", item_id="media-1"):
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


def _receipt_widget_text(screen) -> str:
    receipts = list(screen.query("#library-media-link-receipt"))
    return str(receipts[0].renderable) if receipts else ""


async def _wait_for_receipt(screen, pilot, text: str, app=None) -> None:
    """The receipt widget carries the whole line; the user sees it ellipsised
    to one row beside "Undo link" when the Reader is narrow (the brief's
    Review Focus 5), so the painted check is the workspace-naming head."""
    head = text.split(" · ")[0]

    await _wait_for_condition(
        pilot,
        lambda: _receipt_widget_text(screen) == text and head in _painted(screen),
        message=lambda: (
            f"receipt {text!r} never painted; widget={_receipt_widget_text(screen)!r}; "
            f"notified={_notified(app) if app is not None else '?'!r}"
        ),
    )


async def _wait_for_no_receipt(screen, pilot, why: str) -> None:
    await _wait_for_condition(
        pilot,
        lambda: "Linked to" not in _painted(screen),
        message=why,
    )


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
    AC#2/#6 Undo removes the membership -> AC#4 a failed link refuses in
    plain language, nothing staged -> AC#1 the `c` key takes the same path
    -> Review Focus 2 a workspace switch withholds the stale receipt and the
    next press links into the new workspace, named -> AC#4 no active
    workspace is the global scope (task-15120): stages, links nothing.
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
        assert [(row.workspace_id, row.item_id) for row in rows] == [(RAW_ID, "media-1")]
        app.open_chat_with_handoff.assert_called_once()
        payload = app.open_chat_with_handoff.call_args.args[0]
        assert payload.item_type == "media" and payload.source_id == "media-1"
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
        screen.query_one("#library-media-link-undo", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert _memberships(app) == ()
        await _wait_for_no_receipt(screen, pilot, "receipt still painted after Undo")
        assert not screen.query("#library-media-link-undo")
        gate = screen._library_workspace_depth_state(refresh=True)
        row = next(r for r in gate.source_rows if r.item_id == "media-1")
        assert row.active_context_eligible is False
        assert "Local Default" in row.recovery_copy and RAW_ID not in row.recovery_copy

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
            (RAW_ID, "media-1")
        ]
        assert app.open_chat_with_handoff.call_count == staged_before + 1
        await _wait_for_receipt(screen, pilot, LINKED_RECEIPT, app)

        # -- Review Focus 2: a workspace switched while the item is linked.
        # The stale receipt (and its Undo) is withheld: Undo could only
        # remove a membership of a workspace that is no longer active. The
        # next press links into the NEW active workspace, by name.
        thesis = registry.create_workspace(workspace_id=THESIS_ID, name="Thesis")
        registry.set_active_workspace(thesis.workspace_id)
        screen._invalidate_library_workspace_depth_state()
        screen._sync_library_media_viewer_or_recompose()
        await pilot.pause()
        await pilot.pause()
        await _wait_for_no_receipt(
            screen, pilot, "stale receipt still painted after the workspace switch"
        )
        assert not screen.query("#library-media-link-undo")
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


async def test_search_u_and_reader_c_agree():
    host, app = _media_host()
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "title": "Interview Recording",
                    "snippet": "Matched media · audio",
                    "source_id": "media-1",
                    "provenance": {"source_type": "media"},
                }
            ]
        }
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        screen.query_one("#library-rag-query-input", Input).value = "interview"
        await _wait_for_library_rag_query_ready(screen, pilot, "interview")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-select-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()
        await pilot.pause()
        assert _memberships(app) == (), "control: nothing linked before u"

        await pilot.press("u")
        await pilot.pause()
        await pilot.pause()

        app.open_console_for_live_work.assert_called_once()
        assert app.open_console_for_live_work.call_args.kwargs["title"] == (
            "Interview Recording"
        )
        assert [(row.workspace_id, row.item_id) for row in _memberships(app)] == [
            (RAW_ID, "media-1")
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
