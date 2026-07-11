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

from dataclasses import dataclass
from datetime import date
from pathlib import PurePath
from typing import Mapping

from tldw_chatbook.Library.library_export_scope import ExportScope, export_scope_label

# Exact copy values (binding -- see the F4 plan's Global Constraints).
EXPORT_HEADER_COPY = "Export chatbook"
COUNTING_COPY = "Counting…"
EMPTY_SCOPE_COPY = "Nothing to export in this scope."
MEDIA_QUALITY_HELPER_COPY = "original copies full media files into the zip"
CHOOSE_DESTINATION_COPY = "Choose destination…"
DESTINATION_PLACEHOLDER_COPY = "No destination chosen"
EXPORT_BUTTON_COPY = "Export chatbook"
SERVER_DISABLED_TOOLTIP_COPY = "Export packages local content only."

# The creator's own quality options (thumbnail/compressed/original); default
# is the cheapest one, matching the design spec.
MEDIA_QUALITY_OPTIONS = ("thumbnail", "compressed", "original")
DEFAULT_MEDIA_QUALITY = "thumbnail"

# Scope kinds whose export includes media at all -- everything and
# media-scoped exports show the quality control + helper line;
# conversations/notes-only scopes never touch media, so those rows would
# be dead controls.
_MEDIA_BEARING_SCOPE_KINDS = ("everything", "media")


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
        export_enabled: Whether the "Export chatbook" button is enabled --
            requires counts landed, a non-empty scope, a chosen
            destination, and no export already running.
        show_media_fields: Whether the quality control + its helper line
            should render at all -- only for scopes that can contain
            media (``"everything"``/``"media"``); a conversations- or
            notes-only scope never touches media, so the quality control
            would be a dead knob.
        empty_scope_line: ``EMPTY_SCOPE_COPY`` once counts have landed and
            total to zero, else ``""``.
        overwrite_line: ``"Overwrites {destination filename}"`` when the
            chosen (already ``.zip``-normalized) destination already
            exists on disk, else ``""``. Purely informational -- pressing
            Export proceeds and overwrites; this is not a blocking gate.
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
) -> LibraryExportFormState:
    """Build the export canvas's full display state.

    Args:
        scope: What this export will include.
        counts: The full-query counts for ``scope`` (keys "media"/
            "conversations"/"notes"), or ``None`` while the counts worker
            is still running -- ``counts_loading`` and the ``"Counting…"``
            scope line both derive from this being ``None``.
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
        f"Overwrites {PurePath(destination_clean).name}"
        if destination_clean and destination_exists
        else ""
    )
    export_enabled = (
        not running
        and not counts_loading
        and total > 0
        and bool(destination_clean)
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
    )


def next_media_quality(current: str) -> str:
    """Cycle the media-quality control to its next option, wrapping around.

    Mirrors the media canvas's type-filter cycle button
    (``handle_library_media_type_filter_pressed``) -- a plain ``Select``
    widget did not render reliably in the deployed TUI (see that
    handler's docstring), so every Library form control that picks among
    a small fixed set of options uses this same cycle-button convention
    instead, including this one.

    Args:
        current: The current quality value.

    Returns:
        The next value in ``MEDIA_QUALITY_OPTIONS``, wrapping to the first
        after the last; ``MEDIA_QUALITY_OPTIONS[0]`` when ``current`` is
        not a recognized option.
    """
    try:
        index = MEDIA_QUALITY_OPTIONS.index(current)
    except ValueError:
        return MEDIA_QUALITY_OPTIONS[0]
    return MEDIA_QUALITY_OPTIONS[(index + 1) % len(MEDIA_QUALITY_OPTIONS)]


def normalize_export_destination(path: PurePath) -> PurePath:
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
