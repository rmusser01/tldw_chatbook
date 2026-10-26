"""Save-in-place Persona presentation owned by the incumbent Roleplay screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from ...DB.ChaChaNotes_DB import ConflictError
from ...Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from ...Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane
from ...Widgets.Persona_Widgets.personas_dictionary_detail import (
    PersonasDictionaryDetailWidget,
)
from ...Widgets.Persona_Widgets.personas_library_pane import PersonasLibraryPane
from ...Widgets.Persona_Widgets.personas_lore_detail import PersonasLoreDetailWidget

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


async def finish_dictionary_settings_save(
    screen: PersonasScreen,
    detail: PersonasDictionaryDetailWidget,
    record: dict,
    submitted_snapshot: dict,
    entity_id: str,
) -> None:
    """Acknowledge saved settings and refresh their existing screen presentation.

    Args:
        screen: Incumbent screen that owns selection and optimistic-lock version.
        detail: Mounted detail that owns the submitted and live settings forms.
        record: Actual persisted dictionary returned by the scope service.
        submitted_snapshot: Raw settings captured before awaiting persistence.
        entity_id: Dictionary identity selected for this Save.
    """
    raw_version = record.get("version")
    screen._selected_dictionary_version = (
        int(raw_version) if raw_version is not None else None
    )
    screen.state.has_unsaved_changes = detail.mark_settings_saved(submitted_snapshot)
    screen.state.selected_entity_name = str(record.get("name") or "")
    screen.query_one(PersonasInspectorPane).show_selection(
        name=screen.state.selected_entity_name, kind="dictionary"
    )
    await screen._refresh_dictionary_statistics(record)
    detail.set_status("Saved.")
    screen._update_title()
    await screen._render_dictionary_rows(query=screen.state.search_query)
    library = next(iter(screen.query(PersonasLibraryPane)), None)
    if library is None:
        return
    library.mark_active_row("dictionary", entity_id)
    screen._sync_inspector_console_actions()
    await screen._refresh_dictionary_versions()


async def save_lore_settings(
    screen: PersonasScreen,
    detail: PersonasLoreDetailWidget,
    manager: Any,
    entity_id: str,
    expected_version: int | None,
    payload: dict,
    selected_record: dict | None,
    book_name: str,
    submitted_snapshot: tuple,
) -> None:
    """Persist one admitted Lore settings snapshot and reconcile its own view.

    The screen captures every argument before dispatch. Native persistence drains
    even if Textual cancels the worker; a retired view gets captured-book feedback
    without changing its replacement's selection, controls, or version.
    """
    from ..Screens.personas_screen import _drain_async, _drain_to_thread

    screen._lore_settings_save_started = True

    def is_current() -> bool:
        return (
            screen.is_attached
            and not screen._pruning
            and detail.is_attached
            and not detail._pruning
            and screen.state.active_mode == "lore"
            and screen.state.selected_entity_kind == "lore"
            and screen.state.selected_entity_id == entity_id
            and screen._selected_lore_book is selected_record
            and screen._ready_center_view("lore-detail") is detail
            and getattr(screen.app_instance, "chachanotes_db", None) is manager.db
        )

    def report(status: str, *, failed: bool = False) -> None:
        if is_current():
            detail.set_status(status)
        else:
            screen._notify(
                f"Lore book '{book_name}' (ID {entity_id}): {status}",
                "error" if failed else "information",
            )

    def persist() -> tuple[bool, dict | None]:
        ok = manager.update_world_book(
            int(entity_id),
            name=payload.get("name"),
            description=payload.get("description"),
            scan_depth=payload.get("scan_depth"),
            token_budget=payload.get("token_budget"),
            recursive_scanning=payload.get("recursive_scanning"),
            enabled=payload.get("enabled"),
            expected_version=expected_version,
        )
        if not ok:
            return False, None
        try:
            return True, manager.get_world_book(int(entity_id))
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not reload lore book {entity_id} after save."
            )
            return True, None

    async def reconcile(ok: bool, record: dict | None) -> None:
        nonlocal selected_record
        if not ok:
            report("Save failed: the lore book could not be found.", failed=True)
            return
        if not is_current():
            report("Saved.")
            return
        if record is None:
            report(
                "Saved, but could not reload the lore book. Reselect before saving again."
            )
            return
        raw_version = record.get("version")
        screen._selected_lore_book_version = (
            int(raw_version) if raw_version is not None else None
        )
        screen._selected_lore_book = selected_record = record
        screen.state.selected_entity_name = str(record.get("name") or "")
        screen.query_one(PersonasInspectorPane).show_selection(
            name=screen.state.selected_entity_name, kind="lore"
        )
        screen._update_title()
        await screen._render_lore_rows(query=screen.state.search_query)
        if not is_current():
            report("Saved.")
            return
        screen.query_one(PersonasLibraryPane).mark_active_row("lore", entity_id)
        screen._sync_inspector_console_actions()
        report(
            "Saved. Newer settings edits are not saved."
            if detail.settings_snapshot() != submitted_snapshot
            else "Saved."
        )

    cancellation = None
    try:
        outcome = await _drain_to_thread(
            persist, task_name="personas-lore-save-persist"
        )
        cancellation = outcome.cancellation
        if isinstance(outcome.error, ConflictError):
            report(
                "Save failed: the lore book changed since it was loaded. Reselect and try again.",
                failed=True,
            )
        elif outcome.error is not None:
            logger.opt(exception=outcome.error).warning(
                f"Could not save lore book {entity_id}."
            )
            report(f"Save failed: {outcome.error}", failed=True)
        elif outcome.completed:
            completion = await _drain_async(
                reconcile(*outcome.value), task_name="personas-lore-save-reconcile"
            )
            cancellation = cancellation or completion.cancellation
            if completion.error is not None:
                logger.opt(exception=completion.error).warning(
                    f"Lore book {entity_id} saved but presentation failed."
                )
                screen._notify(
                    f"Lore book '{book_name}' (ID {entity_id}) saved, but the view could not refresh.",
                    "warning",
                )
    finally:
        screen._lore_settings_save_started = False
        screen._lore_settings_save_worker_handle = None
    if cancellation is not None:
        raise cancellation
