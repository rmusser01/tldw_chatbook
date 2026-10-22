"""Save-in-place Persona presentation owned by the incumbent Roleplay screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ...Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from ...Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane

if TYPE_CHECKING:
    from ..Screens.personas_screen import PersonasScreen

logger = logger.bind(module="PersonasScreen")


async def finish_profile_save(
    screen: PersonasScreen,
    saved: dict,
    *,
    source: str | None = None,
    submitted_snapshot: tuple | None = None,
) -> None:
    """Refresh the saved Persona while retaining newer editor drafts.

    Args:
        screen: Incumbent screen that owns the editor and presentation state.
        saved: Actual persisted record, including its identity and version.
        source: Backend that accepted the save.
        submitted_snapshot: Raw form captured by the widget's Save action.
    """
    # Refresh the cached profile list tolerantly even when the user has
    try:
        profiles = await screen.persona_handler.refresh_persona_list(
            raise_on_unavailable=True
        )
    except Exception as exc:
        logger.opt(exception=True).warning(
            "Could not refresh persona profiles after a save."
        )
        screen._profile_lookup_recovery_state = screen._profile_list_recovery_state(exc)
        profiles = []
    else:
        screen._profile_lookup_recovery_state = None
    screen._profiles = [dict(record) for record in (profiles or [])]
    saved_id = str(saved.get("id") or "")
    if saved_id and source == "local":
        await screen._refresh_persona_buddy_lifecycle(saved_id)
    screen._update_purpose_line()
    screen._update_purpose_line()
    if not screen.is_mounted or screen.state.active_mode != "personas":
        # Leave the selection, inspector, and center pane alone.
        return
    screen._edit_mode = "edit"  # create -> edit stays in the editor
    screen._advance_persona_buddy_session()
    screen.state.has_unsaved_changes = False
    screen._set_active_row_unsaved(False)
    name = str(saved.get("name") or "Saved persona")
    screen.state.select_entity(
        entity_kind="persona", entity_id=saved_id, entity_name=name
    )
    inspector = screen.query_one(PersonasInspectorPane)
    revision = saved.get("version")
    inspector.show_selection(
        name=name,
        kind="persona",
        source=screen.persona_handler.current_mode(),
        entity_id=saved_id,
        revision=revision if type(revision) is int else None,
        profile_current=True,
        active=(
            saved.get("is_active", True) is True
            and saved.get("deleted", False) is False
        ),
    )
    inspector.set_unsaved(False)
    inspector.show_validation(())
    screen._sync_inspector_buddy_status()
    # Task 11: keep the read-only policy summary in step with the record.
    inspector.show_policy_rules(saved.get("policy_rules") or [])
    screen._sync_inspector_console_actions()
    # Save-in-place: the returned ``saved`` dict already carries the
    # incremented optimistic-lock version, so the editor (which stays
    # open) re-baselines dirty tracking straight from it - no re-read
    # needed (unlike the character finisher, which reads a stale handler
    # cache and must go back to the DB).
    editor = screen.query_one(PersonaProfileEditorWidget)
    screen.state.has_unsaved_changes = editor.mark_saved(
        saved, submitted_snapshot=submitted_snapshot
    )
    screen._profile_save_inflight = not screen.state.has_unsaved_changes
    screen._set_active_row_unsaved(screen.state.has_unsaved_changes)
    inspector.set_unsaved(screen.state.has_unsaved_changes)
    await screen._render_profile_rows()
    await screen._drain_persona_shared_visual_identity_authoring()
    screen._persona_shared_visual_identity_authority = None
    await screen._discard_persona_visual_authoring_async()
    screen._persona_visual_generation += 1
    shared_snapshot = screen._persona_shared_visual_identity_snapshot(editor)
    if shared_snapshot is not None:
        screen.run_worker(
            screen._configure_persona_shared_visual_identity(shared_snapshot),
            group="personas-shared-visual-identity-load",
            exit_on_error=False,
            exclusive=True,
        )
    visual_snapshot = screen._persona_visual_snapshot(editor)
    if visual_snapshot is not None:
        screen.run_worker(
            screen._configure_persona_visual(visual_snapshot),
            group="personas-persona-visual-load",
            exit_on_error=False,
            exclusive=True,
        )
    screen._show_center("#ccp-persona-editor-view")
    screen._sync_title_and_console_actions()
    screen._notify("Persona saved.", "information")
