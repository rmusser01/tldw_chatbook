"""Source-fenced reusable region preparation; the existing controller owns lifecycle."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from ...Character_Chat.portrait_compositing import (
    COMPONENT_ROLES,
    PortraitCompositionRecipe,
    assemble_components,
    prepared_base_pixel_sha256,
    recover_components,
    validate_recipe,
)
from ...Character_Chat.portrait_performance import (
    PerformanceConflict,
    PerformanceSource,
    PortraitPerformanceRecord,
)
from ...Character_Chat.portrait_performance_authoring import (
    PortraitCompositionCandidate,
    PortraitPerformanceDraft,
    performance_reference,
)
from ...DB.portrait_region_settings import PortraitRegionSettings
from ...Widgets.Persona_Widgets.portrait_region_editor import PortraitRegionEditor

if TYPE_CHECKING:
    from ...DB.ChaChaNotes_DB import CharactersRAGDB
    from ...Image_Generation.capabilities import ResolvedReferenceImage
    from .portrait_performance import PortraitPerformanceController, _Authority


def prepare_regions(
    db: CharactersRAGDB,
    source: PerformanceSource,
    data: bytes,
    record: PortraitPerformanceRecord | None,
) -> tuple[
    ResolvedReferenceImage,
    str,
    PortraitRegionSettings | None,
    int,
    PortraitCompositionCandidate | None,
]:
    """Prepare once and recover provenance only after complete pixel validation."""
    reference = performance_reference(data, source.source_sha256)
    digest = prepared_base_pixel_sha256(reference.content)
    try:
        saved = db.get_character_portrait_regions(
            character_id=source.character_id, expression_key=source.expression_key
        )
        revision = saved.revision if saved else 0
    except ValueError:
        # Unsupported saved algorithms require new confirmed regions, retaining CAS.
        row = db.execute_query(
            "SELECT revision FROM character_portrait_regions WHERE character_id=? AND expression_key=?",
            (source.character_id, source.expression_key),
        ).fetchone()
        saved, revision = None, row[0] if row else 0
    current = None
    candidate = None
    if saved and saved.source == source:
        try:
            validate_recipe(reference.content, saved.recipe)
            current = saved
        except ValueError:
            pass
    if (
        current
        and record
        and record.active
        and record.source == source
        and record.composition_recipe == current.recipe
    ):
        try:
            components = recover_components(
                reference.content, current.recipe, record.frames
            )
            candidate = PortraitCompositionCandidate(
                current.recipe,
                current.revision,
                tuple((role, components[role]) for role in COMPONENT_ROLES),
                record.frames,
            )
        except ValueError:
            pass
    return reference, digest, current, revision, candidate


async def edit_regions(
    controller: PortraitPerformanceController,
    authority: _Authority,
    draft: PortraitPerformanceDraft,
    epoch: int,
) -> bool:
    """Keep the modal bound to the exact prepared draft, editor and profile."""
    screen = controller._screen()
    reference = controller._reference
    if reference is None:
        return False
    modal = controller._region_modal = PortraitRegionEditor(
        reference.content,
        width=reference.width,
        height=reference.height,
        pixel_sha256=controller._base_pixel_sha256,
        recipe=controller._regions.recipe if controller._regions else None,
        save=lambda recipe: controller._save_regions(authority, draft, epoch, recipe),
        is_current=lambda: (
            epoch == controller._epoch
            and controller.draft is draft
            and controller._current(authority)
        ),
    )
    try:
        return await screen.app.push_screen_wait(modal) is True
    finally:
        if controller._region_modal is modal:
            controller._region_modal = None
        if modal.is_mounted:
            modal.retire()


async def save_regions(
    controller: PortraitPerformanceController,
    authority: _Authority,
    draft: PortraitPerformanceDraft,
    epoch: int,
    recipe: PortraitCompositionRecipe,
) -> bool:
    """Save only the captured source/settings revision, replacing changed lineage once."""
    current = lambda: (
        epoch == controller._epoch
        and controller.draft is draft
        and controller._current(authority)
    )
    if not current() or controller._reference is None:
        return False
    reference = controller._reference
    previous = controller._regions
    expected = controller._regions_revision
    changed = (
        previous is None or previous.source != draft.source or previous.recipe != recipe
    )
    view = controller._view()
    if changed:
        speech = controller._speech_task
        if speech is not None:
            speech.cancel()
            await asyncio.gather(speech, return_exceptions=True)
        controller._presentation = None
        if not current():
            return False
    controller._saving = True
    view.set_busy(True, saving=True)
    try:

        def save():
            controller._verify_source(
                authority,
                draft,
                controller._source_request,
                controller._local_authority,
            )
            frames = (
                assemble_components(reference.content, recipe, {}) if changed else ()
            )
            if epoch != controller._epoch or controller.draft is not draft:
                raise PerformanceConflict("portrait_performance_source_changed")
            revision = authority.db.save_character_portrait_regions(
                draft.source, recipe, expected_revision=expected
            )
            return revision, frames

        outcome = await controller._native(authority, save)
        if outcome.error:
            raise outcome.error
        if outcome.cancellation:
            raise outcome.cancellation
        if not outcome.completed or not current():
            return False
        revision, frames = outcome.value
        controller._regions = PortraitRegionSettings(draft.source, recipe, revision)
        controller._regions_revision = revision
        if changed:
            controller._retire_pair_review()
            draft.replace_composition(
                PortraitCompositionCandidate(recipe, revision, (), frames)
            )
            await view.show_draft(draft, current, motion=False)
        view.status(
            "Regions saved. Generate a fresh component batch; currently applied artwork kept."
            if changed
            else "Regions unchanged; saved settings and draft retained."
        )
        return True
    except (ValueError, OSError) as error:
        if current():
            view.status(
                "The source or saved regions changed. Prepare again."
                if isinstance(error, PerformanceConflict)
                else "Regions could not be saved. Current artwork kept."
            )
        return False
    except Exception:  # noqa: BLE001 -- an optional settings failure must preserve the editor and applied artwork
        logger.warning("Portrait region Save failed (category=settings_save_failed).")
        if current():
            view.status(
                "Regions could not be saved. Current artwork kept; retry explicitly."
            )
        return False
    finally:
        controller._saving = False
        view.set_busy(controller._busy)
