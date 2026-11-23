"""The media controller, built on a stand-in host and driven directly (TASK-36952.4).

Owner ruling, 2026-11-23, on how TASK-36952 closes: "Add a direct test, then
close". Every other media test goes through a mounted Library screen. Here
``LibraryMediaController`` is built on a bare Textual screen defined below,
which implements ``LibraryMediaHost``'s four shell-service protocols by
recording each call, so the protocol is shown to be the controller's whole
contract with its host. Production still builds the controller with the
Library screen; this module never imports that screen.
"""
from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from typing import Any, Protocol
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.screen import Screen

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_BROWSE_SKILLS,
)
from tldw_chatbook.UI.Library_Modules.library_media_controller import (
    LibraryMediaController,
    LibraryMediaHost,
)
from tldw_chatbook.UI.Library_Modules.library_media_state import LibraryMediaState
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_NOTES_SOURCE_DATABASE,
)
from tldw_chatbook.Widgets.Library import MediaShellResized


def _recorded(stub):
    """Turn a stub into a member that records its call and returns its answer.

    The stub only names the member and whether it is async; its own
    parameters and body are not used.
    """
    name = stub.__name__
    if inspect.iscoroutinefunction(stub):

        async def member(self, *args, **kwargs):
            return self.record(name, args, kwargs)

    else:

        def member(self, *args, **kwargs):
            return self.record(name, args, kwargs)

    return member


class _StandInHost(Screen):
    """A bare screen implementing ``LibraryMediaHost``'s shell services.

    Each service records its call in ``calls`` and returns
    ``answers[name](*args, **kwargs)``, or None when no answer is set.
    ``shell_state`` backs the controller's accessor and setter keywords.
    """

    def __init__(self) -> None:
        super().__init__()
        self.app_instance = Mock()
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.answers: dict[str, Any] = {}
        self.shell_state: dict[str, Any] = {"media_state": LibraryMediaState()}

    def record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.calls.append((name, args, kwargs))
        answer = self.answers.get(name)
        return answer(*args, **kwargs) if answer is not None else None

    def calls_to(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(args, kwargs) for called, args, kwargs in self.calls if called == name]

    # -- LibraryShellServices
    @_recorded
    def acknowledge_library_destination_change(self): ...
    @_recorded
    def action_library_narrow_stage_return(self): ...
    @_recorded
    async def apply_library_open_item_surface(self): ...
    @_recorded
    def arm_library_list_entry_focus(self): ...
    @_recorded
    def arm_stage_pick_landing(self): ...
    @_recorded
    def close_open_library_choice_strip(self): ...
    @_recorded
    def disarm_library_list_entry_focus(self): ...
    @_recorded
    def focus_library_control(self): ...
    @_recorded
    def focus_library_list_entry(self): ...
    @_recorded
    def focus_library_list_entry_if_current(self): ...
    @_recorded
    def focus_library_rail_action(self): ...
    @_recorded
    def land_stage_pick(self): ...
    @_recorded
    def library_adaptive_reader_allocation_is_current(self): ...
    @_recorded
    def library_entry_route_key(self): ...
    @_recorded
    def library_open_choice_strip(self): ...
    @_recorded
    def library_source_total(self): ...
    @_recorded
    async def open_library_export_canvas(self): ...
    @_recorded
    async def open_library_item_by_id(self): ...
    @_recorded
    def register_library_footer_shortcuts(self): ...
    @_recorded
    def request_library_reader_layout_refresh(self): ...
    @_recorded
    async def run_library_service_call(self): ...
    @_recorded
    def safe_text(self): ...
    @_recorded
    def schedule_library_entry_reconcile(self): ...
    @_recorded
    async def select_library_rail_row(self): ...
    @_recorded
    def source_record_id(self): ...
    @_recorded
    def update_library_ingest_dynamic_regions(self): ...

    # -- LibraryReviewSetServices
    @_recorded
    def active_review_set_banner(self): ...
    @_recorded
    def cancel_pending_review_set_resume(self): ...
    @_recorded
    def notify_review_set(self): ...
    @_recorded
    def review_dismiss_receipt_name(self): ...
    @_recorded
    async def review_dismiss_undo_worker(self): ...
    @_recorded
    async def review_selected_worker(self): ...
    @_recorded
    async def review_set_picker_worker(self): ...
    @_recorded
    def review_set_service(self): ...
    @_recorded
    async def review_these_worker(self): ...
    @_recorded
    def toggle_reviewed_unguarded(self): ...
    @_recorded
    def walk_active_review_set(self): ...

    # -- LibraryReaderLayoutSync
    @_recorded
    def sync_library_conversation_reader_layout_from_shell(self): ...
    @_recorded
    def sync_library_notes_reader_layout_from_shell(self): ...
    @_recorded
    def sync_library_prompts_reader_layout_from_shell(self): ...
    @_recorded
    def sync_library_skills_reader_layout_from_shell(self): ...

    # -- LibraryWorkspaceDepthServices
    @_recorded
    def invalidate_library_workspace_depth_state(self): ...
    @_recorded
    def library_workspace_depth_state(self): ...


def _controller_on(host: _StandInHost) -> LibraryMediaController:
    """Build the controller on ``host``, as the Library screen builds it on itself.

    Each ``<field>_accessor`` keyword reads ``host.shell_state[<field>]`` and
    each ``set_<field>`` keyword writes it, so a keyword added later is
    supplied too.
    """
    state = host.shell_state
    keywords = {
        name: (
            partial(state.__setitem__, name.removeprefix("set_"))
            if name.startswith("set_")
            else partial(state.get, name.removesuffix("_accessor"))
        )
        for name, parameter in inspect.signature(LibraryMediaController).parameters.items()
        if parameter.kind is parameter.KEYWORD_ONLY
    }
    return LibraryMediaController(host, **keywords)


@asynccontextmanager
async def _hosted():
    """Mount a stand-in host in a bare app and build the controller on it."""
    app = App()
    async with app.run_test():
        host = _StandInHost()
        await app.push_screen(host)
        yield host, _controller_on(host)


async def test_the_stand_in_provides_every_member_of_the_host_protocol() -> None:
    """The member list is read from ``LibraryMediaHost`` itself.

    ``__protocol_attrs__`` is the set ``typing`` builds for a protocol over its
    whole MRO (``typing.get_protocol_members`` returns it from Python 3.13), so
    a member added to any of the four service protocols lands in it. Service
    members must be the stand-in's own; the framework members may come from
    Textual's ``Screen``.
    """
    members = set(LibraryMediaHost.__protocol_attrs__)
    services = set().union(
        *(base.__protocol_attrs__ for base in LibraryMediaHost.__bases__ if base is not Protocol)
    )
    async with _hosted() as (host, _controller):
        missing = sorted(
            [name for name in services if name not in vars(_StandInHost)]
            + [name for name in members - services if not hasattr(host, name)]
        )
    assert not missing, f"the stand-in host lacks LibraryMediaHost members: {missing}"


async def test_shell_source_record_id_picks_the_cached_record_an_edit_rewrites() -> None:
    """Shell services: ``source_record_id`` names each cached Media record."""
    edited = {"id": 7, "title": "Old"}
    other = {"id": 8, "title": "Other"}
    async with _hosted() as (host, controller):
        host.answers["source_record_id"] = lambda record: str(record["id"])
        host.shell_state["local_source_records"] = {"media": (edited, other)}
        controller._patch_local_media_record(
            "7", title="New", author="Ann", url="https://example.test/7", keywords=["k"]
        )
    assert host.calls == [("source_record_id", (edited,), {}), ("source_record_id", (other,), {})]
    assert host.shell_state["local_source_records"]["media"] == (
        {"id": 7, "title": "New", "author": "Ann", "url": "https://example.test/7", "keywords": ["k"]},
        other,
    )


async def test_review_set_service_marks_rows_and_caches_the_done_map() -> None:
    """Review-set services: the active set's done marks decorate the rows."""
    review_set = SimpleNamespace(
        items=(
            SimpleNamespace(backing_media_id=7, done=True),
            SimpleNamespace(backing_media_id=8, done=False),
        )
    )
    service = SimpleNamespace(revision=3, get_active_review_set=lambda: review_set)
    async with _hosted() as (host, controller):
        host.answers["review_set_service"] = lambda: service
        rows = controller._decorate_library_media_reviewed(
            ({"backing_media_id": 7}, {"backing_media_id": 8}, {"backing_media_id": 9})
        )
    assert host.calls == [("review_set_service", (), {})]
    assert rows == (
        {"backing_media_id": 7, "reviewed": True},
        {"backing_media_id": 8, "reviewed": False},
        {"backing_media_id": 9, "reviewed": None},
    )
    assert host.shell_state["media_state"].review_done_map_cache == (3, {7: True, 8: False})


@pytest.mark.parametrize(
    ("row_id", "sync"),
    [
        (LIBRARY_ROW_BROWSE_CONVERSATIONS, "sync_library_conversation_reader_layout_from_shell"),
        (LIBRARY_ROW_BROWSE_NOTES, "sync_library_notes_reader_layout_from_shell"),
        (LIBRARY_ROW_BROWSE_PROMPTS, "sync_library_prompts_reader_layout_from_shell"),
        (LIBRARY_ROW_BROWSE_SKILLS, "sync_library_skills_reader_layout_from_shell"),
    ],
)
async def test_a_shell_resize_runs_the_selected_routes_reader_layout_sync(
    row_id: str, sync: str
) -> None:
    """Reader-layout syncs: a resize on a sibling route syncs that route only."""
    async with _hosted() as (host, controller):
        host.shell_state["library_selected_row_id"] = row_id
        host.shell_state["notes_state"] = SimpleNamespace(source=LIBRARY_NOTES_SOURCE_DATABASE)
        event = MediaShellResized()
        controller._resize_library_media_reader_shell(event)
    assert host.calls == [(sync, (), {})]
    assert event._stop_propagation


async def test_workspace_depth_state_refuses_a_hand_off_the_workspace_blocks() -> None:
    """Workspace depth: a blocked depth row refuses "Use in Console"."""
    row = SimpleNamespace(
        item_type="media",
        item_id="7",
        active_context_eligible=False,
        reason_code="no_active_workspace",
        recovery_copy="Select an active workspace before using this item in Console.",
    )
    async with _hosted() as (host, controller):
        host.answers["library_workspace_depth_state"] = lambda refresh=False: SimpleNamespace(
            source_rows=(row,)
        )
        host.shell_state["media_state"].detail = {"id": 7, "title": "Field notes", "content": "Body"}
        controller._open_selected_media_handoff()
    assert host.calls_to("library_workspace_depth_state") == [((), {})]
    host.app_instance.notify.assert_called_once_with(
        row.recovery_copy, severity="warning", markup=False
    )
    host.app_instance.open_chat_with_handoff.assert_not_called()
