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

from tldw_chatbook.Library.library_export_scope import (
    _CONTENTS_PREVIEW_LIMIT,
    ExportScope,
    _count_phrase,
    export_scope_label,
)

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
# task-32232 AC#3: after a failed run the same button is the Retry -- it
# says so, matching the ingest queue's "Retry this batch" grammar rather
# than leaving the user to guess that re-pressing "Export bundle (.zip)"
# is the retry.
EXPORT_RETRY_BUTTON_COPY = "Retry export"
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

MEDIA_QUALITY_OPTIONS = ("thumbnail", "compressed", "original")
# task-32353 AC#1 (critique #10): the default used to be "thumbnail", which
# "keeps a small preview image instead of the full file" -- a silent data
# reduction chosen for someone whose reason for exporting is usually to keep
# the files. A lossy bundle is now something you ask for.
DEFAULT_MEDIA_QUALITY = "original"

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


# task-32353 AC#2: the consequence line's fidelity phrase -- the same three
# options ``_MEDIA_QUALITY_HELPER_COPY`` captions, said in the two words the
# bundle summary has room for. Unknown values degrade to "full files" for the
# same reason the helper copy degrades to "original": it is the conservative
# (most-content) description.
_MEDIA_QUALITY_BUNDLE_COPY: dict[str, str] = {
    "thumbnail": "previews only",
    "compressed": "compressed files",
    "original": "full files",
}


def format_export_bytes(size_bytes: int) -> str:
    """Return a byte count in the Export canvas's own "N KB" spelling.

    One formatter for both halves of the same promise: the pre-write
    estimate on ``consequence_line`` and the post-write receipt in
    ``format_last_export_line``. They round identically so "about 4 KB"
    is never followed by a receipt saying "4.1 KB" for the same bundle.

    Args:
        size_bytes: A non-negative byte count.

    Returns:
        e.g. ``"4 KB"`` -- never "0 KB" for a non-empty bundle (anything
        under 1 KB rounds up to 1).
    """
    # ponytail: KB only, inherited from the receipt's own rounding -- a
    # multi-GB media library's ESTIMATE reads "about 1048576 KB before
    # compression" (the receipt never saw numbers that large). Switch both
    # halves to a unit-stepping formatter (``library_ingest_state.
    # _human_size`` already steps) and re-pin the critique-9 receipt string
    # in the SAME change; splitting them would let estimate and receipt
    # round differently, which is the drift this function exists to prevent.
    return f"{max(1, round(size_bytes / 1024))} KB"


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
        consequence_line: task-32353 AC#2: what pressing Export will
            actually write -- item count, fidelity and estimated size in
            one line above the button; ``""`` while counts are loading.
        contents_lines: task-32353 AC#2: the in-scope items' titles (at
            most ``_CONTENTS_PREVIEW_LIMIT``, then a ``"+ N more"``
            summary), or ``()`` when the scope's items are not
            enumerable up front.
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
    consequence_line: str = ""
    contents_lines: tuple[str, ...] = ()

    @property
    def submit_blocked_reason(self) -> str:
        """Why the Export button is off right now, or ``""`` when it is on.

        task-32362: the blocked button's reason had to reach a mouse
        tooltip to be read at all -- ``"No destination chosen"`` sat
        three rows up. The inline reason under the button and the
        tooltip are THE SAME STRING because both come from
        ``export_button_tooltip``; there is no second sentence to drift.
        """
        return "" if self.export_enabled else export_button_tooltip(self)


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
    titles: tuple[str, ...] = (),
    approx_bytes: int | None = None,
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
        titles: The in-scope items' titles (``ExportPreview.titles``),
            already observed by the counts worker; ``()`` when the
            scope's items are not enumerable up front.
        approx_bytes: Their total stored size in bytes
            (``ExportPreview.approx_bytes``), or ``None`` when unknown --
            ``None`` is rendered as honest copy, never as a zero.

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
    # task-32353 AC#2 (critique #10): the canvas asked for a destination and
    # a name and then wrote a bundle nobody had seen the contents of, at a
    # fidelity chosen by a control rendered at the same weight as "sort".
    # This states the consequence in one line, above the button.
    fidelity = (
        f" · {_MEDIA_QUALITY_BUNDLE_COPY.get(media_quality, 'full files')}"
        if show_media_fields
        else ""
    )
    # "before compression" is load-bearing, not padding: this counts the
    # content going IN, while the receipt after the run stats the zip that
    # came OUT (live check: a 9 KB estimate wrote a 4 KB archive). Without
    # the qualifier the two numbers read as a contradiction.
    size = (
        f" · about {format_export_bytes(approx_bytes)} before compression"
        if approx_bytes is not None
        else " · size known once it runs"
    )
    # A media-only scope counts media items; a mixed scope counts items.
    noun = "media item" if scope.kind == "media" else "item"
    consequence_line = (
        ""
        if counts_loading
        else f"Bundle: {_count_phrase(total, noun)}{fidelity}{size}"
    )
    # The preview lands with the counts, so neither line renders before them.
    # ``titles`` is capped at the limit + 1 by the query, so its length only
    # says WHETHER the list was truncated -- the remainder comes from the
    # counts, which is the only place the true total lives.
    extra = total - _CONTENTS_PREVIEW_LIMIT
    contents_lines = (
        ()
        if counts_loading
        else tuple(titles[:_CONTENTS_PREVIEW_LIMIT])
        + (
            (f"+ {extra} more",)
            if len(titles) > _CONTENTS_PREVIEW_LIMIT and extra > 0
            else ()
        )
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
        consequence_line=consequence_line,
        contents_lines=contents_lines,
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


def format_empty_export_error(requested: int) -> str:
    """Return the "the bundle would have been empty" failure line.

    task-32232: a non-empty selection that collected ZERO items used to
    write a bundle holding README + ``content_items: []`` and report
    success. The creator now refuses to write that archive, and this is
    the canvas copy for it -- it names what the user actually asked for
    (``requested``) so "produced no content" cannot be read as "you
    selected nothing".

    Args:
        requested: How many items the failed run had selected.

    Returns:
        e.g. ``"✗ export produced no content · 3 items were selected"``
        (``"· 1 item was selected"`` for a single-item selection).
    """
    if requested == 1:
        return "✗ export produced no content · 1 item was selected"
    return f"✗ export produced no content · {requested} items were selected"


def format_last_export_line(
    path: str,
    exported_at: float,
    *,
    now: float | None = None,
    item_count: int | None = None,
    size_bytes: int | None = None,
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
        item_count: How many ``content_items`` the WRITTEN archive holds,
            read back from its manifest after the zip landed (task-32232
            AC#4) -- never the requested selection size.
        size_bytes: The written archive's size on disk, likewise stat'd
            from the artifact.

    Returns:
        ``""`` when ``path`` is empty; ``"✓ exported · N items · X KB ·
        <path>"`` once the artifact's own facts are known; otherwise the
        pre-readback fallback ``"Last export: /tmp/out.zip · 2m ago"``
        (a receipt restored from a session that recorded only the path).
    """
    clean_path = str(path or "").strip()
    if not clean_path:
        return ""
    if item_count is not None and size_bytes is not None:
        item_word = "item" if item_count == 1 else "items"
        return (
            f"✓ exported · {item_count} {item_word} · "
            f"{format_export_bytes(size_bytes)} · {clean_path}"
        )
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
