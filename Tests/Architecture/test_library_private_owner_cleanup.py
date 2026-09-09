"""Existing private owners remain live after their screen wrappers are retired."""

import ast
import inspect
from textwrap import dedent
from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.library_notes_session import NoteFlushOutcomeKind
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.parametrize(
    ("caller", "owner", "method"),
    (
        (
            "_flush_library_note_save",
            "_notes_controller",
            "_read_library_note_editor_fields",
        ),
        (
            "_transition_library_notes_presentation",
            "_notes_controller",
            "_sync_library_notes_source_controls",
        ),
        (
            "_return_to_library_database_notes",
            "_notes_controller",
            "_sync_library_notes_source_controls",
        ),
        (
            "_library_note_conflict_snapshot",
            "_notes_controller",
            "_library_note_editor_state",
        ),
        (
            "_library_note_preview_snapshot",
            "_notes_controller",
            "_library_note_editor_state",
        ),
        (
            "_flush_library_note_save",
            "_notes_controller",
            "_gc_pending_blank_note",
        ),
        (
            "_create_library_note",
            "_notes_controller",
            "_reconcile_library_notes_list_canvas",
        ),
        (
            "_delete_library_note_claimed",
            "_notes_controller",
            "_notify_library_note_delete_warning",
        ),
        (
            "compose_content",
            "_conversation_reader_controller",
            "_conversation_reader_list_summary",
        ),
        (
            "_refresh_library_media_detail",
            "_media_controller",
            "_schedule_library_media_image_preview",
        ),
        (
            "handle_library_media_trash_back",
            "_media_controller",
            "_cancel_library_media_trash_delete_confirmation",
        ),
        (
            "_run_library_export_worker",
            "_export_controller",
            "_marshal_library_export_success",
        ),
        (
            "_delete_library_note_claimed",
            "_notes_controller",
            "_remove_library_note_source_record",
        ),
        (
            "_create_library_note",
            "_notes_controller",
            "_append_library_note_source_record",
        ),
        (
            "_flush_library_note_save",
            "_notes_controller",
            "_focus_library_note_validation_field",
        ),
        (
            "_flush_library_note_save",
            "_notes_controller",
            "_route_library_note_validation_field",
        ),
        (
            "_select_library_rail_row_after_source_admission",
            "_ingest_controller",
            "_pause_library_ingest_transient_ui",
        ),
    ),
)
def test_retired_private_calls_read_the_current_controller(
    caller: str, owner: str, method: str
) -> None:
    """Each concrete call reads its current owner instead of a Screen wrapper.

    Args:
        caller: Existing Screen method containing the call.
        owner: Current controller attribute that owns the implementation.
        method: Private implementation whose Screen wrapper is retired.
    """
    implementation = getattr(LibraryScreen, caller)
    if isinstance(implementation, property):
        implementation = implementation.fget
    tree = ast.parse(dedent(inspect.getsource(implementation)))
    calls = [
        node.func
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0]) == f"self.{owner}.{method}"
    assert method not in LibraryScreen.__dict__


@pytest.mark.asyncio
async def test_validation_flush_dispatches_to_owner_replaced_during_save() -> None:
    """A completed flush routes and focuses validation through the live owner."""
    calls: list[tuple[str, str]] = []
    first_calls: list[str] = []
    snapshot = SimpleNamespace(saved_revision=1, body="Draft")
    outcome = SimpleNamespace(
        kind=NoteFlushOutcomeKind.VALIDATION_VETO,
        save_outcome=SimpleNamespace(veto=SimpleNamespace(field="title")),
    )
    owner = SimpleNamespace(
        _route_library_note_validation_field=lambda field: calls.append(
            ("route", field)
        ),
        _focus_library_note_validation_field=lambda field: calls.append(
            ("focus", field)
        ),
    )

    async def flush():
        screen._notes_controller = owner
        return outcome

    screen = SimpleNamespace(
        _notes_controller=SimpleNamespace(
            _route_library_note_validation_field=first_calls.append,
            _focus_library_note_validation_field=first_calls.append,
        ),
        _notes_state=SimpleNamespace(session_blank_id=None),
        _library_note_session=SimpleNamespace(snapshot=snapshot, flush=flush),
        _invalidate_library_note_autosave=lambda: None,
        _update_library_note_meta_static=lambda **kwargs: calls.append(
            ("meta", kwargs["content"])
        ),
    )
    assert await LibraryScreen._flush_library_note_save(screen) is outcome
    assert first_calls == []
    assert calls == [("route", "title"), ("meta", "Draft"), ("focus", "title")]
    assert screen._notes_state.autosave_state == "validation"


@pytest.mark.parametrize("surface", ("history", "memberships"))
def test_prompt_projection_ports_resolve_replaced_owner_at_publish_time(
    surface: str,
) -> None:
    """A producer retained from construction must publish to the current owner.

    Args:
        surface: Prompt history or memberships projection to publish.
    """
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    if surface == "history":
        producer = screen._library_prompt_history_controller
        port = producer._sync_view
        method_name = "_sync_library_prompt_history_region"
    else:
        producer = screen._library_prompt_collections_controller
        port = producer._sync_memberships
        method_name = "_sync_library_prompt_memberships"
    first, second = [], []
    screen._prompts_controller = SimpleNamespace(**{method_name: first.append})
    producer.invalidate()
    initial_state = (
        producer.state if surface == "history" else producer.membership_state
    )
    assert first == [initial_state]

    screen._prompts_controller = SimpleNamespace(**{method_name: second.append})
    producer.invalidate()
    current_state = (
        producer.state if surface == "history" else producer.membership_state
    )
    assert first == [initial_state]
    assert second == [current_state]
    assert (
        producer._sync_view if surface == "history" else producer._sync_memberships
    ) is port


def test_prompt_back_gate_reads_replaced_owner_and_keeps_selection_veto() -> None:
    """The live owner decides editor eligibility; selection still vetoes Back."""
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    screen._prompts_controller = SimpleNamespace(
        _library_prompt_editor_active=lambda: False
    )
    assert screen.check_action("library_prompt_editor_back", ()) is False
    screen._prompts_controller = SimpleNamespace(
        _library_prompt_editor_active=lambda: True
    )
    assert screen.check_action("library_prompt_editor_back", ()) is True
    screen._prompts_state.select_mode = True
    assert screen.check_action("library_prompt_editor_back", ()) is False
