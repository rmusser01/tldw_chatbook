"""Image/Video Settings save acknowledgement and post-save editing context.

The screen continues to own staged values, workers and config authority. These
bounded sibling callbacks acknowledge only the submitted draft, then reload
through the existing off-thread owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.css.query import QueryError
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Input

from ...Widgets.settings_image_gen_panel import ImageGenSettingsPanel
from ...Widgets.settings_video_gen_panel import VideoGenSettingsPanel
from ..Screens.settings_config_models import SettingsCategoryId, SettingsDraft

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


def stage_media_gen_value(
    screen: SettingsScreen,
    category: SettingsCategoryId,
    key: str,
    original: object,
    value: object,
) -> None:
    """Keep explicit field history even when typing returns to its original."""
    draft = screen._settings_drafts.setdefault(
        category, SettingsDraft(category=category)
    )
    has_staged_fields = any(
        staged_key.startswith("field::") for staged_key in draft.values
    )
    draft.set_value(key, original, value)
    # A final typed character can equal the old local value while an
    # earlier save (or Clear) is still writing. Its acknowledgement needs
    # that explicit history; initial equal-valued mount echoes stay clean.
    if not draft.is_dirty and not has_staged_fields:
        screen._settings_drafts.pop(category, None)
    screen._update_draft_status_widgets(category)


def acknowledge_media_gen_save(
    screen: SettingsScreen, category: SettingsCategoryId, submitted: SettingsDraft
) -> None:
    """Advance only submitted baselines; later values remain a draft."""
    draft = screen._settings_drafts.get(category)
    if draft is None:
        draft = SettingsDraft(category=category)
    for key, saved_value in submitted.values.items():
        # Returning a field to its old original can remove the whole
        # draft while the write is pending. It now differs from disk.
        if key.startswith("cleared::") and key not in draft.values:
            field_key = key.replace("cleared::", "field::", 1)
            if field_key in draft.values:
                # Only an explicitly typed field can cancel this Clear.
                # Its old local value now differs from the saved deletion.
                draft.originals[field_key] = ""
            continue
        current = draft.values.get(key, submitted.originals.get(key))
        if current == saved_value:
            draft.values.pop(key, None)
            draft.originals.pop(key, None)
        else:
            draft.set_value(key, saved_value, current)
    if draft.is_dirty:
        screen._settings_drafts[category] = draft
    else:
        screen._settings_drafts.pop(category, None)


async def recompose_media_gen_panel(
    screen: SettingsScreen, panel: ImageGenSettingsPanel | VideoGenSettingsPanel
) -> None:
    """Keep current input context without reviving focus moved elsewhere."""
    collapsed = {node.id: node.collapsed for node in panel.query(Collapsible)}
    scroll_offsets = {
        node.id: node.scroll_offset for node in panel.query(VerticalScroll)
    }
    focused = screen.app.focused
    owns_focus = focused is not None and panel in focused.ancestors
    selection = focused.selection if isinstance(focused, Input) else None
    focus_id = focused.id if owns_focus else None
    if owns_focus:
        # Avoid Textual's removal fallback moving focus to the detail
        # body. A deliberate focus change during the await wins below.
        screen.set_focus(None)
    await panel.recompose()
    for node in panel.query(Collapsible):
        if node.id in collapsed:
            node.collapsed = collapsed[node.id]
    for node in panel.query(VerticalScroll):
        if node.id in scroll_offsets:
            offset = scroll_offsets[node.id]
            node.scroll_to(offset.x, offset.y, animate=False)
    if focus_id is None or screen.app.focused is not None:
        return
    try:
        replacement = panel.query_one(f"#{focus_id}")
    except QueryError:
        return
    # Restoring this editing context must not run Input select-on-focus.
    screen.set_focus(replacement, scroll_visible=False, from_app_focus=True)
    if isinstance(replacement, Input) and selection is not None:
        replacement.selection = selection


async def apply_image_gen_save_result(
    screen: SettingsScreen, saved: bool, warnings: list[str], submitted: SettingsDraft
) -> None:
    if not saved:
        message = "Failed to save Image Gen defaults."
        screen._set_static_text("#settings-imagegen-save-result", message)
        screen.app.notify(message, severity="error")
        return
    acknowledge_media_gen_save(screen, SettingsCategoryId.IMAGE_GENERATION, submitted)
    # Qodo PR #901 fix 3: the save just changed the on-disk truth --
    # invalidate the cached raw-section baseline (see
    # `_image_gen_raw_section_cache`'s docstring).
    screen._image_gen_raw_section_cache = None
    message = "Image Gen defaults saved."
    if warnings:
        message = f"{message} {' '.join(warnings)}"
    try:
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
    except QueryError:
        panel = None
    if panel is not None:
        panel.overlay = screen._image_gen_overlay_values()
        screen._start_image_gen_panel_load(panel.overlay, message)
    screen._update_draft_status_widgets(SettingsCategoryId.IMAGE_GENERATION)
    screen.app.notify(message, severity="warning" if warnings else "information")


async def apply_video_gen_save_result(
    screen: SettingsScreen, saved: bool, warnings: list[str], submitted: SettingsDraft
) -> None:
    if not saved:
        message = "Failed to save Video Gen defaults."
        screen._set_static_text("#settings-videogen-save-result", message)
        screen.app.notify(message, severity="error")
        return
    acknowledge_media_gen_save(screen, SettingsCategoryId.VIDEO_GENERATION, submitted)
    screen._video_gen_raw_section_cache = None
    message = "Video Gen defaults saved."
    if warnings:
        message = f"{message} {' '.join(warnings)}"
    try:
        panel = screen.query_one("#settings-videogen-panel", VideoGenSettingsPanel)
    except QueryError:
        panel = None
    if panel is not None:
        panel.overlay = screen._video_gen_overlay_values()
        screen._start_video_gen_panel_load(panel.overlay, message)
    screen._update_draft_status_widgets(SettingsCategoryId.VIDEO_GENERATION)
    screen.app.notify(message, severity="warning" if warnings else "information")
