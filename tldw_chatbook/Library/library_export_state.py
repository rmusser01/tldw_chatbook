"""Pure display-state for the Library export canvas.

Renders a bulk chatbook-export form from Task 1's ``ExportScope``/
``count_export_scope``/``export_scope_label`` (``library_export_scope.py``)
plus a small local form echo into the immutable
``LibraryExportFormState`` the widget in
``Widgets/Library/library_export_canvas.py`` renders from. Textual-free
(stdlib + ``library_export_scope`` only) so it is unit-testable without
booting the TUI, mirroring ``library_ingest_state.py``.

Every filesystem/DB read this form needs (the counts query, whether the
chosen destination already exists on disk) happens in the screen, off this
pure module -- ``build_library_export_form_state`` only ever receives
already-observed truths (``counts``, ``destination_exists``) as plain
arguments, never performs I/O itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

from tldw_chatbook.Library.library_export_scope import ExportScope, export_scope_label

# Exact copy values. The F4 plan's Global Constraints originally pinned
# EXPORT_HEADER_COPY/EXPORT_BUTTON_COPY to "Export chatbook" -- task-2857
# (Library UAT 2026-08-06, LIB-10) superseded that: "chatbook" appeared
# nowhere else in the UI, so both are "Export bundle (.zip)" now. Do not
# revert to the F4 wording.
EXPORT_HEADER_COPY = "Export bundle (.zip)"
COUNTING_COPY = "Counting…"
EMPTY_SCOPE_COPY = "Nothing to export in this scope."
CHOOSE_DESTINATION_COPY = "Choose destination…"
DESTINATION_PLACEHOLDER_COPY = "No destination chosen"
EXPORT_BUTTON_COPY = "Export bundle (.zip)"
SERVER_DISABLED_TOOLTIP_COPY = "Export packages local content only."

# task-2858 AC#3 (LIB-11): the Export button's tooltip always names either
# what pressing it will do (ready) or the SAME blocker its disabled state
# reflects -- house style is "disabled controls say why" (see the
# select-mode toolbar's export/delete tooltips, F-018). The empty-scope
# case deliberately reuses ``EMPTY_SCOPE_COPY`` verbatim (see
# ``export_button_tooltip``) rather than a second string that could drift
# from the on-canvas line already stating it.
EXPORT_BUTTON_READY_TOOLTIP = "Write the bundle to the chosen destination."
EXPORT_BUTTON_RUNNING_TOOLTIP = "An export is already running."
EXPORT_BUTTON_COUNTING_TOOLTIP = "Waiting for item counts before exporting."
EXPORT_BUTTON_NO_DESTINATION_TOOLTIP = "Choose a destination before exporting."

# The creator's own quality options (thumbnail/compressed/original); default
# is the cheapest one, matching the design spec.
MEDIA_QUALITY_OPTIONS = ("thumbnail", "compressed", "original")
DEFAULT_MEDIA_QUALITY = "thumbnail"

# task-2859 item 3: the helper line used to be one FIXED sentence describing
# "original" quality ("original copies full media files into the zip"),
# shown verbatim no matter which option the cycle button actually had
# selected -- so picking "thumbnail ▸" was captioned with a description of
# "original". Each option now gets its own honest caption.
_MEDIA_QUALITY_HELPER_COPY: dict[str, str] = {
    "thumbnail": "keeps a small preview image instead of the full file",
    "compressed": "shrinks media files before adding them to the zip",
    "original": "copies full media files into the zip",
}


def media_quality_helper_copy(media_quality: str) -> str:
    """Return the helper line describing ``media_quality``'s actual effect.

    Args:
        media_quality: The quality control's current value (one of
            ``MEDIA_QUALITY_OPTIONS``).

    Returns:
        The matching one-line caption, or the "original" caption for an
        unrecognized value (the safest/most conservative description).
    """
    return _MEDIA_QUALITY_HELPER_COPY.get(
        media_quality, _MEDIA_QUALITY_HELPER_COPY["original"]
    )


# Scope kinds whose export includes media at all -- everything and
# media-scoped exports show the quality control + helper line;
# conversations/notes-only scopes never touch media, so those rows would
# be dead controls.
_MEDIA_BEARING_SCOPE_KINDS = ("everything", "media")

# ``format_last_export_line``'s relative-age bucket thresholds, named so the
# 60/3600/86400 in that function read as units, not magic numbers.
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400


def default_export_name(today: date | None = None) -> str:
    """Return the form's prefilled export name, stamped with today's date.

    Args:
        today: The date to stamp with; defaults to the local
            ``date.today()``. Exposed as a parameter so callers (and
            tests) can pin the stamp instead of depending on wall-clock
            time.

    Returns:
        ``"Library export YYYY-MM-DD"``.
    """
    stamp = today if today is not None else date.today()
    return f"Library export {stamp.isoformat()}"


@dataclass(frozen=True)
class LibraryExportFormState:
    """Full display state for the Library export canvas.

    The first eleven fields are the Task 2/3 contract (Task 3's execution
    worker and button handler read ``scope``/``name``/``description``/
    ``media_quality``/``destination`` and drive ``running``/
    ``status_line``/``error_line``) -- their names and types are load-
    bearing and must not change. The remaining fields are canvas-render-
    only conveniences derived here so the widget never has to duplicate
    this module's business rules.

    Attributes:
        scope: What this export will include.
        scope_line: The scope summary line -- ``COUNTING_COPY`` while
            ``counts_loading``, else ``export_scope_label(scope, counts)``.
        counts_loading: Whether the full-query counts worker has not yet
            landed a result for the current scope.
        name: The export's display name, as typed (prefilled via
            ``default_export_name`` when the form opens).
        description: Optional description, as typed.
        media_quality: One of ``MEDIA_QUALITY_OPTIONS``.
        destination: The chosen, ``.zip``-normalized destination path, or
            ``""`` until one is picked.
        running: Whether an export is currently executing (Task 3).
        status_line: A quiet in-progress line (Task 3, e.g. ``"Exporting…
            (N items)"``); empty when not running.
        error_line: The last export failure's message, or ``""``.
        export_enabled: Whether the "Export bundle (.zip)" button is enabled --
            requires counts landed, a non-empty scope, a chosen
            destination, and no export already running.
        show_media_fields: Whether the quality control + its helper line
            should render at all -- only for scopes that can contain
            media (``"everything"``/``"media"``); a conversations-,
            notes-, or Prompts-only scope never touches media, so the
            quality control would be a dead knob.
        empty_scope_line: ``EMPTY_SCOPE_COPY`` once counts have landed and
            total to zero, else ``""``.
        overwrite_line: ``"Overwrites {destination filename}"`` when the
            chosen (already ``.zip``-normalized) destination already
            exists on disk, else ``""``. Purely informational -- pressing
            Export proceeds and overwrites; this is not a blocking gate.
        last_export_line: task-2858 AC#3 (LIB-12): the durable
            ``"Last export: <path> · <relative time>"`` receipt for the
            most recent successful export THIS SESSION, or ``""`` before
            any export has completed. Built by ``format_last_export_line``
            from screen-owned state that survives
            ``_reset_library_export_transient_state`` -- unlike every
            other field above, this is NOT derived from the current
            scope/form.
    """

    scope: ExportScope
    scope_line: str
    counts_loading: bool
    name: str
    description: str
    media_quality: str
    destination: str
    running: bool
    status_line: str
    error_line: str
    export_enabled: bool
    show_media_fields: bool = True
    empty_scope_line: str = ""
    overwrite_line: str = ""
    last_export_line: str = ""
    # task-14902: True while the quality chooser's direct-pick strip
    # renders below its (still-visible) opener button.
    quality_choices_visible: bool = False


def build_library_export_form_state(
    *,
    scope: ExportScope,
    counts: Mapping[str, int] | None,
    name: str,
    description: str,
    media_quality: str,
    destination: str,
    destination_exists: bool = False,
    running: bool = False,
    status_line: str = "",
    error_line: str = "",
    last_export_line: str = "",
    quality_choices_visible: bool = False,
) -> LibraryExportFormState:
    """Build the export canvas's full display state.

    Args:
        scope: What this export will include.
        counts: The full-query counts for ``scope`` (keys "media"/
            "conversations"/"notes"/"prompts"), or ``None`` while the
            counts worker is still running -- ``counts_loading`` and the
            ``"Counting…"`` scope line both derive from this being ``None``.
        name: The export name field's current text.
        description: The description field's current text.
        media_quality: The quality control's current value.
        destination: The chosen destination path (already ``.zip``-
            normalized by the caller), or ``""``.
        destination_exists: Whether ``destination`` already exists on
            disk -- an already-observed filesystem truth the caller
            supplies; this function performs no I/O of its own.
        running: Whether an export is currently executing.
        status_line: The in-progress status line (Task 3).
        error_line: The last failure's message, if any.
        last_export_line: The durable receipt line (task-2858 AC#3,
            LIB-12), already formatted by ``format_last_export_line`` --
            this function only passes it through.

    Returns:
        The canvas's full display state.
    """
    counts_loading = counts is None
    resolved_counts = counts or {}
    total = sum(resolved_counts.values())
    scope_line = (
        COUNTING_COPY if counts_loading else export_scope_label(scope, resolved_counts)
    )
    show_media_fields = scope.kind in _MEDIA_BEARING_SCOPE_KINDS
    empty_scope_line = EMPTY_SCOPE_COPY if not counts_loading and total == 0 else ""
    destination_clean = str(destination or "").strip()
    overwrite_line = (
        f"Overwrites {Path(destination_clean).name}"
        if destination_clean and destination_exists
        else ""
    )
    export_enabled = (
        not running and not counts_loading and total > 0 and bool(destination_clean)
    )
    return LibraryExportFormState(
        scope=scope,
        scope_line=scope_line,
        counts_loading=counts_loading,
        name=name,
        description=description,
        media_quality=media_quality,
        destination=destination,
        running=running,
        status_line=status_line,
        error_line=error_line,
        export_enabled=export_enabled,
        show_media_fields=show_media_fields,
        empty_scope_line=empty_scope_line,
        overwrite_line=overwrite_line,
        last_export_line=last_export_line,
        quality_choices_visible=quality_choices_visible,
    )


def export_button_tooltip(state: LibraryExportFormState) -> str:
    """Return the Export button's tooltip: why it's disabled, or the ready hint.

    task-2858 AC#3 (LIB-11): "disabled controls say why" -- mirrors
    ``export_enabled``'s own predicate order (running -> counts still
    loading -> empty scope -> no destination) so the tooltip always names
    the ACTUAL current blocker instead of a generic "can't click this".
    The empty-scope branch reuses ``state.empty_scope_line`` (== exactly
    ``EMPTY_SCOPE_COPY`` whenever counts have landed at zero) verbatim, so
    the tooltip can never drift from the on-canvas line stating the same
    fact.

    Args:
        state: The canvas's full display state.

    Returns:
        A non-empty tooltip string in every case -- the button always
        explains either what it will do or what is blocking it.
    """
    if state.export_enabled:
        return EXPORT_BUTTON_READY_TOOLTIP
    if state.running:
        return EXPORT_BUTTON_RUNNING_TOOLTIP
    if state.counts_loading:
        return EXPORT_BUTTON_COUNTING_TOOLTIP
    if state.empty_scope_line:
        return state.empty_scope_line
    if not state.destination.strip():
        return EXPORT_BUTTON_NO_DESTINATION_TOOLTIP
    # Defensive only: export_enabled mirrors this exact predicate chain,
    # so every False case is covered above -- this is unreachable in
    # practice, but a silently blank tooltip would be worse than a
    # slightly-generic fallback if the two predicates ever drift.
    return EMPTY_SCOPE_COPY


def format_last_export_line(
    path: str, exported_at: float, *, now: float | None = None
) -> str:
    """Return the durable "Last export: <path> · <relative time>" receipt line.

    task-2858 AC#3 (LIB-12): a successful export used to leave the canvas
    pixel-identical -- no on-screen sign the zip was ever written. The
    caller (the screen) records ``path``/``exported_at`` in state that
    survives ``_reset_library_export_transient_state``, so this line
    reappears every time the export canvas is (re)composed for the rest
    of the session, not just immediately after the run that produced it.

    Args:
        path: The destination path a successful export wrote to this
            session, or ``""`` if nothing has been exported yet.
        exported_at: ``time.time()`` epoch seconds when that export
            completed.
        now: ``time.time()`` epoch seconds to measure "ago" against;
            defaults to the real current time. Exposed so tests can pin
            it instead of depending on wall-clock time (mirrors
            ``default_export_name``'s ``today`` parameter).

    Returns:
        ``""`` when ``path`` is empty; otherwise the formatted receipt,
        e.g. ``"Last export: /tmp/out.zip · 2m ago"``.
    """
    clean_path = str(path or "").strip()
    if not clean_path:
        return ""
    current = time.time() if now is None else now
    elapsed = max(0.0, current - exported_at)
    if elapsed < _SECONDS_PER_MINUTE:
        relative = "just now"
    elif elapsed < _SECONDS_PER_HOUR:
        relative = f"{int(elapsed // _SECONDS_PER_MINUTE)}m ago"
    elif elapsed < _SECONDS_PER_DAY:
        relative = f"{int(elapsed // _SECONDS_PER_HOUR)}h ago"
    else:
        relative = f"{int(elapsed // _SECONDS_PER_DAY)}d ago"
    return f"Last export: {clean_path} · {relative}"


def normalize_export_destination(path: Path) -> Path:
    """Normalize a chosen destination path's suffix to ``.zip``.

    The chatbook creator silently coerces whatever suffix it's given, so
    normalizing here -- *before* any overwrite confirmation is computed --
    ensures the path the user confirms overwriting is the actual path
    that gets written, not the raw picked one (design spec, "Export
    form").

    Args:
        path: The raw path returned by the ``FileSave`` dialog.

    Returns:
        ``path`` unchanged if it already ends in ``.zip`` (case-
        insensitive), else ``path`` with its suffix replaced by
        ``.zip``.
    """
    if path.suffix.lower() == ".zip":
        return path
    return path.with_suffix(".zip")
