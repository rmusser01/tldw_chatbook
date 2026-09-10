"""First-use Buddy conversion workflow owned by the Personas workbench."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...Utils.paths import get_user_data_dir


def capture_buddy_import_guard(screen: Any) -> Callable[[], bool]:
    """Pin the local destination before the shared Characters file picker."""
    app = screen.app_instance
    db = getattr(app, "chachanotes_db", None)
    scope = getattr(app, "character_persona_scope_service", None)
    local = getattr(scope, "local_service", None)
    root = Path(get_user_data_dir())
    selection = (
        screen.state.active_mode,
        screen.state.selected_entity_id,
        screen._persona_visual_generation,
    )

    def current() -> bool:
        return bool(
            screen.is_mounted
            and db is not None
            and local is not None
            and getattr(app, "chachanotes_db", None) is db
            and getattr(app, "character_persona_scope_service", None) is scope
            and getattr(scope, "local_service", None) is local
            and Path(get_user_data_dir()) == root
            and screen._local_character_actions_allowed()
            and (
                screen.state.active_mode,
                screen.state.selected_entity_id,
                screen._persona_visual_generation,
            )
            == selection
        )

    return current


async def review_buddy_character(
    screen: Any, *, archive: bool, archive_path: Path | None = None
) -> None:
    """Release the dialog slot even when destination capture fails."""
    try:
        await _review_buddy_character(
            screen, archive=archive, archive_path=archive_path
        )
    except (ValueError, OSError, RuntimeError):
        screen._notify("Buddy source or destination is unavailable.", "error")
    finally:
        screen._io_dialog_active = False


async def _review_buddy_character(
    screen: Any, *, archive: bool, archive_path: Path | None = None
) -> None:
    """Review one immutable source against the captured local destination."""
    from ...Character_Chat.buddy_conversion import suggest_buddy_mappings
    from ...Persona_Visual.repository import PersonaVisualRepository
    from ...Persona_Visual.snapshot import read_buddy_archive, read_saved_buddy
    from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters
    from ...Widgets.Persona_Widgets.buddy_character_review import (
        BuddyCharacterReviewDialog,
    )

    app = screen.app_instance
    db = getattr(app, "chachanotes_db", None)
    scope = getattr(app, "character_persona_scope_service", None)
    service = getattr(scope, "local_service", None)
    root = Path(get_user_data_dir())
    source = screen._persona_visual_authoring
    source_authority = source.snapshot if source is not None else None
    generation = screen._persona_visual_generation
    selected_id = screen.state.selected_entity_id
    mode = screen.state.active_mode

    def destination_current() -> bool:
        return bool(
            screen.is_mounted
            and db is not None
            and service is not None
            and getattr(app, "chachanotes_db", None) is db
            and getattr(app, "character_persona_scope_service", None) is scope
            and getattr(scope, "local_service", None) is service
            and Path(get_user_data_dir()) == root
            and screen._local_character_actions_allowed()
        )

    def current() -> bool:
        return bool(
            destination_current()
            and screen.state.active_mode == mode
            and screen.state.selected_entity_id == selected_id
            and screen._persona_visual_generation == generation
            and (
                archive
                or (
                    source_authority is not None
                    and screen._persona_visual_snapshot_is_current(source_authority)
                    and not screen._persona_visual_has_unsaved_authoring()
                )
            )
        )

    try:
        if not current():
            screen._notify(
                "Save or cancel Buddy changes before creating a character. A local destination is required.",
                "warning",
            )
            return
        if archive:
            path = archive_path
            if path is None:
                path = await screen.app.push_screen_wait(
                    EnhancedFileOpen(
                        title="Create character from Buddy archive",
                        filters=Filters(
                            (
                                "Buddy packs",
                                lambda value: (
                                    value.suffix.lower() == ".tldw-persona-vpack"
                                ),
                            )
                        ),
                        context="buddy_character_import",
                    )
                )
            if not path or not current():
                return
            snapshot = await asyncio.to_thread(read_buddy_archive, Path(path))
        else:
            snapshot = await asyncio.to_thread(
                read_saved_buddy,
                PersonaVisualRepository(db),
                source_authority.persona_id,
                root,
            )
        if not current():
            screen._notify(
                "Buddy or destination changed. Start a fresh review.", "warning"
            )
            return
        rows = await asyncio.to_thread(suggest_buddy_mappings, snapshot)
        if not await asyncio.to_thread(snapshot.is_current) or not current():
            screen._notify(
                "Buddy or destination changed. Start a fresh review.", "warning"
            )
            return
        created = await screen.app.push_screen_wait(
            BuddyCharacterReviewDialog(
                snapshot,
                rows,
                db=db,
                local_service=service,
                profile_root=root,
                authority_guard=current,
                config=getattr(app, "app_config", {}) or {},
            )
        )
        if created is None:
            return
        if not current():
            screen._notify(
                "Character was created in the original local destination. Reopen Characters to find it.",
                "information",
            )
            return
        await screen._refresh_after_actor_pack_activation(created.result)
        if not created.open_console or not current():
            return

        # Use the existing draft guard before the explicit navigation action.
        async def open_created() -> None:
            if not destination_current():
                return
            await screen._apply_mode("characters")
            if not destination_current():
                return
            await screen._select_character(created.result.local_actor_id, created.name)
            # load_character schedules the card worker; its cache must be hydrated
            # before the established handoff action reads the full character.
            for worker in tuple(screen.workers):
                if worker.group == "ccp-load-character":
                    await worker.wait()
            if (
                not destination_current()
                or screen.state.selected_entity_kind != "character"
                or screen.state.selected_entity_id != created.result.local_actor_id
            ):
                return
            await screen._attach_selection_to_console(intent="start_chat")

        await screen._run_guarded(open_created)
    except (ValueError, OSError, RuntimeError) as exc:
        screen._notify(f"Could not review Buddy: {exc}", "error")
