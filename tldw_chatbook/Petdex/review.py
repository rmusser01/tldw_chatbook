"""Disposable Petdex preparation and guarded native draft handoff."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any


@dataclass(frozen=True)
class PetdexReviewedArchive:
    """Pinned bytes approved for an unpublished native import."""

    archive: bytes
    source: Any


async def drain_thread(function: Callable, *args: Any, **kwargs: Any) -> Any:
    """Drain owned background work before cancellation releases its resources."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class PetdexPreviewCodecUnavailable(ValueError):
    """The installed Pillow cannot encode the animated review format."""


def preview_state(
    source: Any,
    inspection: Any,
    state: Any,
    *,
    animate: bool = True,
) -> bytes:
    """Encode review frames, closing all decoded images even when encoding fails."""
    from PIL import Image, features

    if animate and not features.check("webp"):
        raise PetdexPreviewCodecUnavailable("WebP preview encoder unavailable")
    with ExitStack() as resources:
        sheet = resources.enter_context(
            closing(Image.open(BytesIO(source.image_bytes)))
        )
        frames = []
        for index in range(state.frames if animate else 1):
            cropped = sheet.crop(
                (
                    index * inspection.cell_width,
                    state.row * inspection.cell_height,
                    (index + 1) * inspection.cell_width,
                    (state.row + 1) * inspection.cell_height,
                )
            )
            resources.callback(cropped.close)
            frame = resources.enter_context(closing(cropped.convert("RGBA")))
            frame.thumbnail((192, 208), Image.Resampling.NEAREST)
            frames.append(frame)
        with BytesIO() as output:
            if animate:
                duration, remainder = divmod(state.duration_ms, state.frames)
                frames[0].save(
                    output,
                    format="WEBP",
                    save_all=True,
                    append_images=frames[1:],
                    lossless=True,
                    duration=[duration + (i < remainder) for i in range(state.frames)],
                    loop=0 if state.loop else 1,
                )
            else:
                frames[0].save(output, format="PNG")
            return output.getvalue()


async def review_petdex_import(screen: Any) -> None:
    """Capture a saved local destination before opening the source dialog."""
    try:
        from ..Utils.paths import get_user_data_dir
        from ..Widgets.Persona_Widgets.petdex_import_review import (
            PetdexImportReviewDialog,
        )

        state = screen._persona_visual_authoring
        draft = state.draft if state else None
        root = Path(get_user_data_dir())
        app = screen.app_instance
        db = getattr(app, "chachanotes_db", None)
        scope = getattr(app, "character_persona_scope_service", None)
        service = getattr(scope, "local_service", None)

        def current() -> bool:
            return bool(
                state is not None
                and screen._persona_visual_authoring is state
                and state.draft is draft
                and not state.dirty
                and state.snapshot.db is db
                and state.snapshot.local_service is service
                and screen._persona_visual_snapshot_is_current(state.snapshot)
                and getattr(app, "chachanotes_db", None) is db
                and getattr(app, "character_persona_scope_service", None) is scope
                and getattr(scope, "local_service", None) is service
                and Path(get_user_data_dir()) == root
            )

        if not current() or not await drain_thread(
            screen._persona_visual_authority_guard, state.snapshot
        ):
            screen._notify(
                "Save or cancel visual changes and select a saved active local Persona.",
                "warning",
            )
            return
        if not current():
            return
        result = await screen.app.push_screen_wait(
            PetdexImportReviewDialog(
                authority_guard=current, config=getattr(app, "app_config", {}) or {}
            )
        )
        if result is None or not current():
            return
        if not await drain_thread(result.source.is_current) or not current():
            screen._notify(
                "Petdex source or destination changed. Start a fresh review.", "warning"
            )
            return
        with TemporaryDirectory(prefix="tldw-petdex-") as folder:
            archive = Path(folder) / "reviewed.tldw-persona-vpack"
            await drain_thread(archive.write_bytes, result.archive)
            if not current():
                return
            await screen._import_persona_visual_from_path(
                str(archive),
                source_guard=lambda: (
                    result.source.is_current()
                    and screen._persona_visual_authority_guard(state.snapshot)
                ),
                destination_guard=current,
            )
    except (ValueError, OSError, RuntimeError):
        screen._notify("Could not import Petdex source. Start a fresh review.", "error")
    finally:
        screen._io_dialog_active = False


def export_target_identity(target: Path) -> tuple | None:
    """Capture the selected output identity without following a file symlink."""
    if target.is_symlink():
        raise ValueError("Export target cannot be a symlink.")
    try:
        value = target.stat()
    except FileNotFoundError:
        return None
    if not target.is_file():
        raise ValueError("Export target must be a file.")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def write_native_export(
    target: Path,
    data: bytes,
    *,
    expected_identity: tuple | None,
    authority_guard: Callable[[], bool],
) -> None:
    """Publish complete bytes atomically, preserving any older output on failure."""
    temporary = None
    try:
        with NamedTemporaryFile(
            prefix=".tldw-export-", dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if not authority_guard() or export_target_identity(target) != expected_identity:
            raise ValueError("Export destination changed.")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def export_native_buddy(screen: Any) -> None:
    """Expose the notice-preserving native exporter for a saved local Buddy."""
    try:
        from ..Persona_Visual.export import export_persona_visual_archive
        from ..Persona_Visual.repository import PersonaVisualRepository
        from ..Utils.path_validation import validate_path_simple
        from ..Utils.paths import get_user_data_dir
        from ..Widgets.enhanced_file_picker import EnhancedFileSave

        state = screen._persona_visual_authoring
        root = Path(get_user_data_dir())
        app = screen.app_instance
        db = getattr(app, "chachanotes_db", None)
        scope = getattr(app, "character_persona_scope_service", None)
        service = getattr(scope, "local_service", None)

        def current() -> bool:
            return bool(
                state is not None
                and screen._persona_visual_authoring is state
                and not state.dirty
                and screen._persona_visual_snapshot_is_current(state.snapshot)
                and state.snapshot.db is db
                and state.snapshot.local_service is service
                and getattr(app, "chachanotes_db", None) is db
                and getattr(app, "character_persona_scope_service", None) is scope
                and getattr(scope, "local_service", None) is service
                and Path(get_user_data_dir()) == root
            )

        def write_current() -> bool:
            return bool(
                screen.app.call_from_thread(current)
                and screen._persona_visual_authority_guard(state.snapshot)
            )

        if not current() or not await drain_thread(
            screen._persona_visual_authority_guard, state.snapshot
        ):
            return
        if not current():
            return
        target = await screen.app.push_screen_wait(
            EnhancedFileSave(
                title="Export saved Buddy pack",
                default_filename="buddy.tldw-persona-vpack",
                context="persona_visual_native_export",
            )
        )
        if not target or not current():
            return
        validated = validate_path_simple(str(target))
        if not validated or Path(validated).suffix.lower() != ".tldw-persona-vpack":
            raise ValueError("Choose a .tldw-persona-vpack file.")
        target = Path(validated)
        target_identity = await drain_thread(export_target_identity, target)
        if not current():
            return
        data = await drain_thread(
            export_persona_visual_archive,
            PersonaVisualRepository(state.snapshot.db),
            state.snapshot.persona_id,
            root,
        )
        if not current():
            return
        await drain_thread(
            write_native_export,
            target,
            data,
            expected_identity=target_identity,
            authority_guard=write_current,
        )
        screen._notify(
            "Saved Buddy pack exported with its artwork notices.", "information"
        )
    except (ValueError, OSError, RuntimeError):
        screen._notify("Could not export the saved Buddy pack.", "error")
    finally:
        screen._io_dialog_active = False
