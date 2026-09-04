"""Pure display-state for the Library ingest canvas.

Renders the app-level Library ingest job registry (``library_ingest_jobs.py``)
plus a small local form echo into the immutable state
``LibraryIngestCanvas`` (the widget in ``Widgets/Library/library_ingest_canvas.py``)
renders from. Textual-free (stdlib only) so it is unit-testable without
booting the TUI, mirroring ``library_notes_sync_state.py``.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import PurePath


from collections.abc import Mapping
from typing import Any, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import (
    format_console_relative_age as format_batch_relative_age,
)
from tldw_chatbook.Local_Ingestion.ingest_parse_progress import (
    INGEST_PARSE_PROGRESS_MESSAGE_MAX_CHARS,
)
from tldw_chatbook.Library.ingest_capabilities import (
    MULTI_PAGE_SCRAPE_METHODS,
    _is_installed as _dependency_installed,
    classify_missing_features,
    generic_option_default,
    get_capabilities,
    list_type_groups,
)
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import (
    DEFAULT_CHUNK_SIZE,
    INGEST_DUPLICATE_PROGRESS_PREFIX,
    IngestJobState,
    LibraryIngestJob,
)
from tldw_chatbook.Library.server_ingest_request import server_ingest_refusal


def _generic_default(name: str, fallback: Any) -> Any:
    """Return the ``generic`` group's declared default for ``name``.

    The capability schema is the single source of ingest option defaults.
    This form echo used to hard-code its own, which disagreed with it
    (``analyze``, ``chunk_size``) and with the other ingest surface
    (``chunk``) -- three answers to the same question, so a user's actual
    defaults depended on which screen they happened to open. (task-3301)
    Delegates to ``ingest_capabilities.generic_option_default``, the shared
    accessor every consumer of these defaults now goes through.
    """
    return generic_option_default(name, fallback)

# Exact copy values (binding -- see the L3b plan's Global Constraints).
INGEST_HEADER_COPY = "Import media"
# Retired: the old scope-warning line ("ingest runs on Local") existed to say
# that ingest ignored the Library's browse scope. Ingest never keys off browse
# scope now -- it has its own explicit target -- so the line is replaced by one
# that names that target. Kept as a name for any external reference.
SERVER_QUIET_LINE_COPY = "ingest runs on Local"
INGEST_TARGET_LOCAL_COPY = "Imports run on this machine."
INGEST_TARGET_SERVER_COPY = "Imports run on the server."
INGEST_SERVER_NEEDS_SERVER_MODE_COPY = (
    "Imports run on this machine. Switch the Library to server mode to "
    "import on the server."
)
MEDIA_DB_UNAVAILABLE_COPY = "Media database is unavailable."
INGEST_UNAVAILABLE_COPY = "Import is unavailable in this runtime."
QUEUE_HEADING_COPY = "Queue"
QUEUE_EMPTY_COPY = "No import jobs yet."
# (task-2130) After a session with activity the old line was a lie.
QUEUE_EMPTY_AFTER_ACTIVITY_COPY = "Queue is empty."
# (task-3305, MI-12) The surface accepts URLs, so the nudge must say so.
START_QUIET_LINE_COPY = "Enter a file path or URL to start."
SUPPORTED_FORMATS_COPY = (
    "Supported: PDF documents, Word/Office documents, audio/video files, "
    "e-books, images, plain text files, web pages (by URL)."
)
#: (task-14823) The gate reason for a folder that holds no files at all.
#: Distinct from the all-unsupported sentence on purpose: the recovery is
#: "put files in it / pick another folder", not "these formats aren't
#: supported". Mirrors the not-found branch's shape -- a named reason at
#: the control plus the way out.
INGEST_EMPTY_SELECTION_COPY = (
    "This folder is empty — there's nothing to import. Choose a folder "
    "with files, or a single file."
)


def active_ingest_start_confirm_line(
    *,
    active_source_count: int,
    is_folder: bool,
    tooling_affected_count: int,
) -> str:
    """Return the bounded inline consent copy for active-source matches.

    Args:
        active_source_count: Number of sources that match active ingest jobs.
        is_folder: Whether the submitted source is a folder batch.
        tooling_affected_count: Number of sources affected by a tooling warning.

    Returns:
        One-line confirmation instruction, or an empty string when no active
        source match exists.
    """
    if active_source_count and tooling_affected_count:
        return (
            f"Import active; {tooling_affected_count} may fail. "
            "Start again to queue."
        )
    if active_source_count and is_folder:
        noun = "file" if active_source_count == 1 else "files"
        return (
            f"{active_source_count} active {noun}. "
            "Start again to queue all."
        )
    if active_source_count:
        return "Import active. Start again to queue a duplicate."
    return ""

#: (xhigh review of task-14823) The gate reason for a folder that holds
#: entries the scan passed over -- symlinks, dot-entries, unreadable
#: subfolders. ``total_files == 0`` covers this case too, and the sentence
#: above was asserted about it: a folder of symlinked media was told it
#: was EMPTY, and task-14823's submit gate then refused it outright, so
#: the wrong diagnosis became a dead end. The recovery is different as
#: well ("import the file itself" works; "put files in it" is nonsense
#: for a folder that already has some), so it gets its own sentence.
INGEST_UNSCANNABLE_SELECTION_COPY = (
    "Nothing in this folder could be scanned — {count} {noun} skipped: "
    "folder imports pass over hidden files, links, and folders they "
    "can't read. Import a file directly, or choose another folder."
)


def ingest_unscannable_selection_copy(skipped_entries: int) -> str:
    """Render :data:`INGEST_UNSCANNABLE_SELECTION_COPY` for a skip count."""
    return INGEST_UNSCANNABLE_SELECTION_COPY.format(
        count=skipped_entries,
        noun="entry was" if skipped_entries == 1 else "entries were",
    )


# (task-3303 AC5) The local article extractor is single-page: the multi-page
# scrape methods (sitemap/url_level/recursive) are honored only by the server
# clip path, so a local "sitemap" selection used to silently import ONE page.
WEB_LOCAL_SINGLE_PAGE_NOTE = (
    "Multi-page fetch runs on the server — this local import fetches one "
    "page."
)


def build_web_scope_note(
    ingest_backend: str, web_options: Mapping[str, Any]
) -> str:
    """Return the local single-page honesty note for the web options panel.

    (task-3303 AC5) ``scrape_method``/``max_pages``/``max_depth`` are honored
    only by the server clip path; the local article path fetches exactly one
    page. When the ingest targets the local backend and a multi-page method
    is selected, the panel must say so at the control instead of letting the
    run silently import a single page.

    Args:
        ingest_backend: Where a new ingest will run (``"local"``/``"server"``,
            the canvas state's ``ingest_backend``).
        web_options: The ``web`` group's current option values from the form
            echo (missing keys fall back to the schema defaults).

    Returns:
        :data:`WEB_LOCAL_SINGLE_PAGE_NOTE` when the note applies, else ``""``.
    """
    if str(ingest_backend or "local").strip().lower() == "server":
        return ""
    fields = {f.name: f for f in get_capabilities("web").fields}
    default_method = getattr(fields.get("scrape_method"), "default", "individual")
    method = str(
        (web_options or {}).get("scrape_method") or default_method
    ).strip()
    return WEB_LOCAL_SINGLE_PAGE_NOTE if method in MULTI_PAGE_SCRAPE_METHODS else ""


#: (task-3306) The trim bounds travel VERBATIM to ffmpeg (-ss/-to/-t) and
#: yt-dlp's postprocessor args, whose time-duration syntax is plain seconds
#: (optionally fractional) or [HH:]MM:SS[.fraction] with minutes/seconds
#: below 60. Anything else fails the job only at run time, so the form
#: gates the format up front.
_TRIM_TIME_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?|(?:\d+:)?[0-5]?\d:[0-5]?\d(?:\.\d+)?)$"
)

#: Audio/video fields validated as trim timestamps (see above).
_TRIM_TIME_FIELDS = frozenset({"start_time", "end_time"})


def validate_ingest_option_value(field: Any, value: Any) -> str:
    """Validation message for one option value, or ``""`` when valid.

    (task-2130) Shared by the state gate and the canvas's inline per-field
    messages so the two can never disagree. ``number`` fields and the
    audio/video trim timestamps (task-3306) have a wrong shape; other
    types are constrained by their widgets. The chunk-size bounds mirror
    ``clamp_chunk_size``'s submit-time clamp (Qodo round: a value the UI
    blessed must not be silently rewritten at submit).

    Args:
        field: The ``OptionField`` schema entry for the value.
        value: The raw form value (display text for Inputs).

    Returns:
        A human-readable problem statement, or ``""`` when the value is
        acceptable to every downstream consumer.
    """
    if getattr(field, "name", "") in _TRIM_TIME_FIELDS:
        text = str(value).strip()
        if not text or _TRIM_TIME_RE.match(text):
            return ""
        return f"{field.label} must be HH:MM:SS or seconds."
    if getattr(field, "type", "") != "number":
        return ""
    text = str(value).strip()
    try:
        number = int(text)
    except (TypeError, ValueError):
        return f"{field.label} must be a whole number."
    name = getattr(field, "name", "")
    if name == "chunk_size":
        if not (MIN_CHUNK_SIZE <= number <= MAX_CHUNK_SIZE):
            return (
                f"{field.label} must be between {MIN_CHUNK_SIZE} and "
                f"{MAX_CHUNK_SIZE}."
            )
        return ""
    minimum = 0 if name == "chunk_overlap" else 1
    if number < minimum:
        return f"{field.label} must be at least {minimum}."
    return ""


def collect_ingest_option_errors(
    type_options: Mapping[str, Mapping[str, Any]],
    groups: Sequence[str] | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Collect validation problems for the option values a user can see.

    (task-2130 Qodo round) Scoped to the RENDERED groups and to fields
    whose ``enabled_when`` gate is currently satisfied: a stale persisted
    value in a panel that is not on screen (or in a field the UI shows
    disabled) must not silently block Start with nothing visible to fix.
    ``depends_on`` (missing tooling) fields are likewise skipped -- the
    widget is disabled, so its value cannot be corrected in place.

    Args:
        type_options: Per-group option values from the form echo.
        groups: The groups whose panels are rendered; ``None`` validates
            every known group (used by tests and non-UI callers).

    Returns:
        ``(group, field name, message)`` tuples, in schema order.
    """
    errors: list[tuple[str, str, str]] = []
    group_names = list_type_groups() if groups is None else groups
    for group in group_names:
        cap = get_capabilities(group)
        values = type_options.get(group, {}) or {}
        fields_by_name = {f.name: f for f in cap.fields}
        for option_field in cap.fields:
            if option_field.depends_on is not None and not _dependency_installed(
                option_field.depends_on
            ):
                continue
            if option_field.enabled_when is not None:
                gate = fields_by_name.get(option_field.enabled_when)
                gate_value = values.get(
                    option_field.enabled_when,
                    gate.default if gate is not None else False,
                )
                if option_field.enabled_when_values:
                    if gate_value not in option_field.enabled_when_values:
                        continue
                elif not bool(gate_value):
                    continue
            message = validate_ingest_option_value(
                option_field,
                values.get(option_field.name, option_field.default),
            )
            if message:
                errors.append((group, option_field.name, message))
    return tuple(errors)

# First-visit orientation. Shown only while the form is untouched, so it fills
# the otherwise-blank pane a new user lands on without ever competing with a
# real pre-flight summary.
INGEST_INTRO_WHAT_COPY = (
    "Import a file, a whole folder, or a URL. Supported: {types}."
)
INGEST_INTRO_NEXT_COPY = (
    "Imported items are searchable in your Library and can be used as "
    "context in chat."
)

# Re-exported from library_ingest_jobs.py (the lowest-level pure module in
# the Library ingest stack) rather than redefined here -- kept as a module
# attribute of this file too (not just imported-and-discarded) since
# existing consumers/tests import ``DEFAULT_CHUNK_SIZE`` from
# ``library_ingest_state``.
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 5000

# Queue row state glyphs (binding).
_GLYPH_ACTIVE = "●"  # "●" -- queued, parsing, or writing
_GLYPH_DONE = "✓"  # "✓"
#: (task-2837) A dedup match is a distinct OUTCOME, not a quieter
#: import -- the two used to be byte-identical rows.
_GLYPH_MATCHED = "≡"
_GLYPH_FAILED = "✗"  # "✗"
_GLYPH_SKIPPED = "○"  # neutral: never attempted (task-2220)
_GLYPH_CANCELLED = "⊘"  # "⊘" -- stopped deliberately, not an error

# L4: the marker `local_file_ingestion.py`'s "Unsupported file type" error
# copy uses to separate the offending extension from its own supported-list
# tail -- shared here so the queue row's ``short_error`` split can never
# drift out of sync with that error string's exact punctuation.
_SUPPORTED_TYPES_ERROR_MARKER = " Supported types:"

# (task-2015) The pipeline historically wrapped stage errors in
# ``Failed to <verb> <type> file:`` at each layer, producing chains like
# ``Failed to ingest pdf file: Failed to process pdf file: PDF Extraction
# Error.``. Only an outer prefix immediately followed by another
# ``Failed to`` is dropped -- a single prefix carries real information and
# passes through untouched.
_NESTED_FAILURE_PREFIX_RE = re.compile(
    r"^Failed to \w+(?: [\w.+-]+)? file: (?=Failed to )"
)

# (task-3312 #2) ``EgressBlockedError``'s message opens with exactly this
# phrase (``Utils/egress.py``); it is the one marker every egress-refused
# fetch carries regardless of which pipeline wrapper re-raised it.
_EGRESS_BLOCKED_MARKER = "Egress blocked"

#: (task-3312 #2) Plain-language receipt for an egress-blocked URL. The raw
#: policy text ("URL blocked by egress policy (SSRF guard): Egress blocked
#: (private) for http://… [remedy: add the host to [web_security]
#: allowed_hosts in config.toml, or set [web_security] enabled = false]")
#: leaked a markup escape into the queue row (a literal ``\[web_security]``
#: -- ``rich.markup.escape`` and Textual's content markup disagree about
#: which brackets needed escaping) and clipped mid-sentence at
#: "config.toml," (live 2026-08-08). One complete sentence in the
#: pre-flight lines' register (task-3305), bracket-free, remedy intact.
INGEST_EGRESS_BLOCKED_COPY = (
    "URL blocked — your web-security settings don't allow fetching this "
    "address. To allow it, add the host to allowed_hosts under "
    "web_security in config.toml."
)

#: (xhigh review round) ``EgressBlockedError`` renders the refused target
#: as a credential- and query-free origin ("http://host:port") right after
#: its reason slug. Pulling it back out is what lets the receipt NAME the
#: host: task-3312's fixed sentence said "this address" and never which
#: one, so a queue of refusals read as N identical rows and the expanded
#: details could not recover the target either.
_EGRESS_ORIGIN_RE = re.compile(
    r"Egress blocked \([^)]*\) for (?P<origin>\S+)"
)


def egress_blocked_receipt(error: str) -> str:
    """Plain-language egress refusal that names the host it refused.

    Keeps task-3312's register exactly -- one complete sentence, no policy
    jargon, no bracketed config-key syntax for the renderer to eat -- and
    adds the one fact the user needs to act: WHICH address was refused.

    An origin that cannot be recovered, or one whose rendering carries
    square brackets (a bracketed IPv6 literal, e.g. ``http://[::1]:8000``
    -- the exact character class that leaked a stray ``\\[`` into a live
    queue row), falls back to the host-less sentence rather than shipping
    markup-hostile text.

    Args:
        error: The raw (already unwrapped) job error text.

    Returns:
        The receipt to show on the queue row and Home's failed-item line.
    """
    match = _EGRESS_ORIGIN_RE.search(error)
    origin = match.group("origin").rstrip(".,:;") if match else ""
    if not origin or "[" in origin or "]" in origin or origin.startswith("<"):
        return INGEST_EGRESS_BLOCKED_COPY
    return (
        f"URL blocked — your web-security settings don't allow fetching "
        f"{origin}. To allow it, add the host to allowed_hosts under "
        "web_security in config.toml."
    )


def short_ingest_error(error: str) -> str:
    """Return the short (queue-row) form of an ingest job's error message.

    Drops the trailing ``" Supported types: ..."`` tail that
    ``local_file_ingestion.py``'s "Unsupported file type" error carries --
    that tail is dropped from the queue-row summary so it is not repeated
    on every failure surface. An error without that exact marker passes
    through whole. An egress-blocked URL failure is rewritten by
    :func:`egress_blocked_receipt` -- the raw policy text is log material,
    not a receipt (task-3312 #2) -- which keeps that plain-language
    register while naming the refused host (xhigh review round: the fixed
    sentence said "this address" and never which one).

    Single source of truth for BOTH failure-reason surfaces: the Library
    ingest queue row (``_build_queue_row``) and Home's failed-item canvas
    line (``active_work_adapter._local_ingest_job_items``) call this same
    helper, so the two can never drift apart (F1b whole-wave review).

    Args:
        error: The raw ``LibraryIngestJob.error`` text.

    Returns:
        The error up to (excluding) the supported-types marker, right-
        stripped; the whole error when the marker is absent; the plain
        egress receipt when the egress-blocked marker is present.
    """
    unwrapped = unwrap_ingest_error(error)
    if _EGRESS_BLOCKED_MARKER in unwrapped:
        return egress_blocked_receipt(unwrapped)
    return unwrapped.split(_SUPPORTED_TYPES_ERROR_MARKER)[0].rstrip()


def unwrap_ingest_error(error: str) -> str:
    """Collapse nested ``Failed to <verb> <type> file:`` wrapper chains.

    Shared by ``short_ingest_error`` (queue row) and the expanded
    detail lines (task-2043), so neither surface ever shows
    "Failed to ingest pdf file: Failed to process pdf file: …".

    Args:
        error: The raw error text.

    Returns:
        The text with at most one leading ``Failed to … file:`` prefix.
    """
    while True:
        unwrapped = _NESTED_FAILURE_PREFIX_RE.sub("", error, count=1)
        if unwrapped == error:
            return error
        error = unwrapped


def _strip_basename_echo(detail: str, basename: str) -> str:
    """Drop a leading repeat of the row's own basename from its detail.

    (task-3305) The pipeline's error copy often opens with the filename
    (``"empty.txt is empty; there was nothing to ingest."``), which the
    queue row already carries -- ``"✗ failed · empty.txt · empty.txt is
    empty…"`` read as a stutter. Only an exact leading repeat is dropped;
    anything else passes through whole.

    Args:
        detail: The short error text destined for the row line.
        basename: The row's displayed basename.

    Returns:
        ``detail`` without the leading basename echo (and any separator
        punctuation that followed it), or ``detail`` unchanged.
    """
    if not basename or not detail.startswith(basename):
        return detail
    rest = detail[len(basename):]
    # Only a WORD-BOUNDARY echo counts: "report.txt is empty" stutters,
    # but "report.txt.orig could not be read" names a sibling artifact and
    # must pass through whole (xhigh review of task-3305).
    if rest and not rest[0].isspace() and rest[0] not in ":-·—,":
        return detail
    trimmed = rest.lstrip().lstrip(":-·—,").lstrip()
    return trimmed if trimmed else detail


def _retry_suffix(job: LibraryIngestJob) -> str:
    """Return a `` · retry {n}`` suffix once a job has been requeued.

    Mirrors ``active_work_adapter._ingest_retry_suffix`` -- kept as a
    separate (Library-side) copy rather than a shared import so this
    module's Textual-free, importable-in-isolation contract (see the module
    docstring) never has to reach into ``Home`` (the dependency runs the
    other way: ``Home`` already imports from ``Library``).
    """
    return f" · attempt {job.retry_count + 1}" if job.retry_count else ""

# Human-readable (singular, plural) labels for pre-flight type groups.
# ``unsupported`` is popped into ``unsupported_files`` before this mapping is
# consulted, so it is intentionally absent here.
_TYPE_GROUP_LABELS: dict[str, tuple[str, str]] = {
    "pdf": ("PDF document", "PDF documents"),
    # (task-3303) .doc/.docx/.odt/.rtf have their own group now -- the
    # pre-flight used to count them as "plain text files".
    "document": ("Word/Office document", "Word/Office documents"),
    "audio_video": ("audio/video file", "audio/video files"),
    "ebook": ("e-book", "e-books"),
    # (task-3307) Raster images have their own group now -- the pre-flight
    # used to drop them in the unsupported bucket.
    "image": ("image", "images"),
    "generic": ("plain text file", "plain text files"),
    # (task-3305, MI-18) The fallback pluralised the group id -- a URL
    # selection read "1 web".
    "web": ("web page", "web pages"),
}


def _human_size(size_bytes: int) -> str:
    """Return a compact human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    # Sizes above ~1 PB: divide once more so the value is actually in petabytes.
    value /= 1024
    return f"{value:.1f} PB"


def build_type_breakdown_line(type_groups: dict[str, list[str]]) -> str:
    """Build a human-readable file/type breakdown line.

    Args:
        type_groups: Mapping from capability group to the paths assigned to
            that group, as returned by ``PreflightResult.type_groups`` (after
            any ``unsupported`` group has been popped).

    Returns:
        A comma-joined summary such as ``"1 PDF document, 2 audio/video files"``,
        or an empty string when ``type_groups`` is empty.
    """
    if not type_groups:
        return ""
    parts: list[str] = []
    for group, paths in type_groups.items():
        count = len(paths)
        singular, plural = _TYPE_GROUP_LABELS.get(group, (group, f"{group}s"))
        label = singular if count == 1 else plural
        parts.append(f"{count} {label}")
    return ", ".join(parts)


def build_estimate_line(total_files: int, total_size: int, truncated: bool) -> str:
    """Build a lightweight file-count + size estimate line.

    Args:
        total_files: Number of files discovered.
        total_size: Sum of file sizes in bytes.
        truncated: Whether the directory scan reached its limit.

    Returns:
        ``"{n} file(s) · {size}"`` with an optional truncation note.
    """
    if total_files <= 0:
        return "0 files"
    noun = "file" if total_files == 1 else "files"
    line = f"{total_files} {noun} · {_human_size(total_size)}"
    if truncated:
        line += " · more files not shown"
    return line


def build_intro_lines() -> tuple[str, ...]:
    """Return the first-visit orientation lines for the ingest canvas.

    The supported-type list is derived from the same labels the pre-flight
    breakdown uses, so what the empty state promises and what the analysis
    reports cannot drift apart.
    """
    types = ", ".join(plural for _singular, plural in _TYPE_GROUP_LABELS.values())
    return (
        INGEST_INTRO_WHAT_COPY.format(types=types),
        INGEST_INTRO_NEXT_COPY,
    )


def _advisory_line(warning: Mapping[str, Any] | None) -> str:
    """Render a pre-flight note that names no missing feature.

    (review round) These are advisories, not tooling gaps -- the URL
    probe's "Could not check the link" is the live example. They must not
    borrow :func:`build_warning_lines`' "X isn't installed -- needed for
    Y" shape, which turned that note into a sentence claiming a component
    called "Could not check the link" was missing.

    Args:
        warning: A pre-flight warning mapping carrying no ``feature``.

    Returns:
        ``"Label — hint"``, either half alone, or ``""`` when neither is
        present.
    """
    label = str((warning or {}).get("label") or "").strip()
    hint = str((warning or {}).get("hint") or "").strip()
    if label and hint:
        return f"{label} — {hint}"
    return label or hint


def build_warning_lines(warnings: list[dict[str, Any]]) -> list[str]:
    """Build human-readable warning lines from pre-flight warning dicts.

    Args:
        warnings: List of warning dictionaries, typically from
            ``PreflightResult.warnings``. Expected keys: ``label``, ``hint``,
            and optionally ``command``.

    Returns:
        A list of display strings such as ``"PDF processing isn't installed —
        needed for PDF ingestion. Install it with: pip install -e \\".[pdf]\\""``.

        Each line names the missing thing, what it costs the user, and the
        command that fixes it. The old shape pasted a label in front of a hint
        that already contained it, producing "PDF processing: PDF processing is
        unavailable: PDF ingestion." -- so the "needed for" clause is dropped
        whenever it would merely repeat the label.
    """
    lines: list[str] = []
    for warning in warnings:
        label = (warning.get("label") or "").strip()
        hint = (warning.get("hint") or "").strip()
        command = (warning.get("command") or "").strip()

        # ``hint`` is normally a noun phrase ("PDF ingestion"), but a caller
        # may hand over a whole sentence; trimming the terminator keeps the
        # composed line from ending in "..".
        hint = hint.rstrip(".").strip()

        if label:
            sentence = f"{label} isn't installed"
            if hint and hint.casefold() != label.casefold():
                sentence += f" — needed for {hint}"
            sentence += "."
        elif hint:
            sentence = f"{hint}."
        else:
            lines.append(str(warning))
            continue

        if command:
            sentence += f" Install it with: {command}"
        lines.append(sentence)
    return lines


#: (task-3313) The "Retry this batch" affordance's resting label.
LIBRARY_INGEST_RETRY_LABEL = "Retry this batch"
#: (xhigh review + live-verify round) Its armed label. A re-stage replaces
#: path/title/author/keywords/options wholesale with no undo, so when it
#: would discard work the user entered since the submit it takes the
#: incumbent two-press consent -- and the affordance itself carries the
#: pending state, because the ``r`` accelerator has no other surface.
LIBRARY_INGEST_RETRY_CONFIRM_LABEL = "Press again to replace form"


def library_ingest_retry_label(retry_confirm_armed: bool) -> str:
    """Return the retry affordance's label for the current consent state."""
    return (
        LIBRARY_INGEST_RETRY_CONFIRM_LABEL
        if retry_confirm_armed
        else LIBRARY_INGEST_RETRY_LABEL
    )


def library_ingest_retry_available(
    jobs: Sequence[LibraryIngestJob],
    *,
    last_submission_available: bool,
) -> bool:
    """Whether "Retry this batch" is offered at all -- key AND button.

    (task-3313; consolidated in the xhigh review + live-verify round) The
    affordance appears once a submission exists this session AND the queue
    has settled: an active job means that submission has not reached a
    terminal state yet, and re-staging mid-run invites a duplicate batch.

    This is the ONE predicate. ``build_library_ingest_state`` derives
    ``show_retry_last`` from it and ``LibraryScreen.check_action`` gates
    the ``r`` binding on it. They used to state the rule separately and
    the screen's copy omitted the settled-queue half, so mid-run the key
    stayed live exactly while the button was deliberately hidden -- one
    keystroke away from the duplicate batch the button was hidden to
    prevent.

    Args:
        jobs: The registry's current job snapshot.
        last_submission_available: Whether a last-submission snapshot
            exists for this session.

    Returns:
        ``True`` when the affordance (button and key alike) should be
        offered.
    """
    return bool(last_submission_available) and not any(
        job.state
        in (
            IngestJobState.QUEUED,
            IngestJobState.PARSING,
            IngestJobState.WRITING,
        )
        for job in jobs
    )


def preflight_install_commands(warnings: list[dict[str, Any]]) -> tuple[str, ...]:
    """The distinct install commands behind a pre-flight's warnings, in order.

    (task-3304, MI-17) Feeds the summary's copy affordance: the composed
    warning prose embeds the command mid-sentence, which is unreadable at
    a canvas edge and uncopyable everywhere -- the guardrail modal used to
    be the only place with a copy button. Several features can resolve to
    the same extra (e.g. soundfile + scipy -> ``.[audio]``), so commands
    are de-duplicated while keeping first-seen order.

    Args:
        warnings: Pre-flight warning dicts (``PreflightResult.warnings``).

    Returns:
        Unique, non-empty ``command`` strings in first-appearance order.
    """
    commands: dict[str, None] = {}
    for warning in warnings:
        command = str(warning.get("command") or "").strip()
        if command:
            commands.setdefault(command, None)
    return tuple(commands)


@dataclass(frozen=True)
class IngestForecast:
    """What a staged selection will actually do, computed exactly once.

    (task-14820) The commit line and the inline consent line used to be
    derived independently -- ``will_import = supported_total - will_match``
    for one, ``count_warning_affected_files`` for the other -- and
    disagreed on screen: live saw ``15 will import`` two rows above
    ``⚠ … 7 files may fail``, delivering ``8 imported · 5 skipped · 8
    failed``. Both lines are now rendered FROM this one object
    (:func:`forecast_summary_line`, :func:`forecast_consent_line`), so
    they are structurally incapable of stating different numbers for the
    same selection.

    Attributes:
        will_import: Files expected to land in the Library as new rows.
        will_match: Files the duplicate probe expects to match instead of
            re-import (a floor when ``match_capped``).
        match_capped: Whether ``will_match`` is a capped floor, not a total.
        will_skip: Unsupported files -- never attempted by the pipeline.
            Always ``0`` under ``targets_server``: the server path skips
            nothing, it fails what it cannot map (``will_fail_refused``).
        will_fail_refused: Files the TARGETED backend refuses outright.
            Server-only today, and a different set from ``will_skip``:
            the server additionally refuses types it has no media type
            for (images), while a page is not refused at all because it
            is routed to the clipper. See
            :func:`~tldw_chatbook.Library.server_ingest_request.server_ingest_refusal`.
        will_fail_tooling: Files whose type group has an unmet REQUIRED
            feature. The pre-flight already warned about that exact
            dependency, so promising these as imports was the defect.
        will_fail_empty: 0-byte files, which fail identically every time.
            True on BOTH backends because both refuse them in code this
            process runs: locally the parse chain raises
            ``EmptySourceIngestError`` before any write, and (task-14910)
            a server submission refuses one at
            :func:`~tldw_chatbook.Library.server_ingest_request.empty_source_refusal`
            rather than sending it and letting a server this process
            cannot inspect decide -- which is what this count used to
            quietly assume.
        at_risk: Files that are still forecast to import but whose group
            has an unmet OPTIONAL feature -- degraded, not doomed.
        tooling_groups: The type groups blocked by a missing required
            feature, in pre-flight order.
        staged_total: Every file in the selection, whatever its fate.
            Carried here rather than re-derived by a caller, because a
            second computation of the same number is exactly the defect
            this object exists to remove -- and because which buckets a
            caller would have to sum is a property of the pre-flight, not
            of this dataclass (``analyze_path`` lifts 0-byte files OUT of
            ``type_groups``, so ``will_fail_empty`` is disjoint from the
            group counts; a pre-flight that did not would silently make a
            summing caller wrong).
        targets_server: Whether this forecast describes a SERVER run. A
            server import never loads a local parser, so the local tooling
            gaps that drive ``will_fail_tooling``/``at_risk`` say nothing
            about it -- both are ``0`` here, and ``will_import`` is the
            count that will be SENT rather than a promise about what the
            server can do with them (its capabilities are not knowable
            from this process; see task-3309 on unverified forwarded
            extras). The renderers say so in words.
    """

    will_import: int = 0
    will_match: int = 0
    match_capped: bool = False
    will_skip: int = 0
    will_fail_refused: int = 0
    will_fail_tooling: int = 0
    will_fail_empty: int = 0
    at_risk: int = 0
    tooling_groups: tuple[str, ...] = ()
    staged_total: int = 0
    targets_server: bool = False

    @property
    def will_fail(self) -> int:
        """Total forecast failures, whatever the reason."""
        return (
            self.will_fail_refused
            + self.will_fail_tooling
            + self.will_fail_empty
        )

    @property
    def consent_affected(self) -> int:
        """Files the tooling warnings put at risk -- doomed or degraded.

        The inline consent line's blast radius, derived here rather than
        recomputed by the caller (that second computation IS task-14820).
        """
        return self.will_fail_tooling + self.at_risk


def build_ingest_forecast(
    preflight: PreflightResult | None, *, targets_server: bool = False
) -> IngestForecast | None:
    """Forecast a staged selection's outcomes from ONE computation.

    The single source of truth for the commit-point forecast and the
    inline consent line. Keyed off the pre-flight's OWN warnings rather
    than a fresh probe of this process, so what the forecast counts and
    what the warning wall says are the same fact: a group whose REQUIRED
    feature was warned about cannot run at all (those files are certain
    failures), while a group with only an OPTIONAL feature warned about
    still imports, degraded (those files are ``at_risk``).

    Those tooling gaps are LOCAL facts, so they bear only on a LOCAL run.
    (xhigh review round) Subtracting them unconditionally made server mode
    worse than the defect this function was written to fix: five .mp3 with
    no local audio extra forecast "0 will import · 5 will fail (need
    tooling)" for a batch the server would have transcribed in full --
    ``_submit_server_ingest_job`` never touches a local parser. Under
    ``targets_server`` the local gaps are dropped entirely and no claim is
    made about the server's own tooling, because this process cannot know
    it (task-3309: forwarded extras are unverified). What IS knowable is
    what gets sent, and that is what the renderers say.

    Args:
        preflight: The active pre-flight result, or ``None``.
        targets_server: Whether the staged run will be submitted to the
            server rather than parsed on this machine.

    Returns:
        The forecast, or ``None`` when there is no pre-flight result or
        the result carries errors (a path error owns that state -- the
        error and its recovery affordance render instead).
    """
    if preflight is None or preflight.errors:
        return None
    warned = {
        str(warning.get("feature") or "").strip()
        for warning in preflight.warnings
        if isinstance(warning, Mapping)
    } - {""}
    type_groups = {
        group: files
        for group, files in preflight.type_groups.items()
        if group != "unsupported"
    }
    will_import = 0
    fail_refused = 0
    fail_tooling = 0
    at_risk = 0
    tooling_groups: list[str] = []
    if targets_server:
        # (task-14827) Ask the backend that will actually run this. The
        # local verdict answers a different question: a raster image is a
        # perfectly good LOCAL import (group ``image``, OCR) that the
        # server has no media type for, and an unclassifiable file is
        # SKIPPED locally but recorded as a permanent FAILURE by
        # ``_submit_server_ingest_job``. Per FILE, not per group, because
        # the refusal is a property of the source, not of the group.
        for files in preflight.type_groups.values():
            for path in files:
                if server_ingest_refusal(str(path)) is None:
                    will_import += 1
                else:
                    fail_refused += 1
    else:
        for group, files in type_groups.items():
            count = len(files)
            required_missing, optional_missing = classify_missing_features(
                group, warned
            )
            if required_missing:
                fail_tooling += count
                tooling_groups.append(group)
                continue
            will_import += count
            if optional_missing:
                at_risk += count
    # (task-2223) The duplicate probe is a capped best-effort count over
    # the read≈parse groups; subtract it from the files that would
    # otherwise import, never from the ones already forecast to fail.
    will_match = min(
        int(getattr(preflight, "already_in_library", 0) or 0), will_import
    )
    return IngestForecast(
        will_import=will_import - will_match,
        will_match=will_match,
        match_capped=bool(
            getattr(preflight, "already_in_library_capped", False)
        ),
        # (task-14827) Nothing is skipped on the server path -- the
        # unsupported group's files are inside ``fail_refused`` above,
        # counted by asking the server mapping rather than the local one.
        will_skip=(
            0
            if targets_server
            else len(preflight.type_groups.get("unsupported", ()))
        ),
        will_fail_refused=fail_refused,
        will_fail_tooling=fail_tooling,
        will_fail_empty=len(getattr(preflight, "empty_files", ()) or ()),
        at_risk=at_risk,
        tooling_groups=tuple(tooling_groups),
        staged_total=int(getattr(preflight, "total_files", 0) or 0),
        targets_server=bool(targets_server),
    )


#: (xhigh review round) The server-mode tail. A server import's outcome
#: turns on tooling installed on the SERVER, which this process cannot
#: inspect -- task-3309 is open precisely because forwarded extras go
#: unverified. Saying so is the only honest alternative to asserting
#: either extreme ("5 will import" promises what we cannot know; "5 will
#: fail (need tooling)" condemns a run on someone else's machine using
#: this machine's inventory).
INGEST_SERVER_TOOLING_UNKNOWN_COPY = "server tooling isn't checked from here"

#: (task-14827) Why the failures in a SERVER forecast are failures. It
#: names the backend doing the refusing on purpose: half of these files
#: (raster images) import perfectly well on THIS machine, so the local
#: vocabulary -- "unsupported", "will skip", "no handler for this format"
#: -- would tell a user their file is unreadable when what is true is
#: that this particular destination will not take it.
INGEST_SERVER_REFUSED_COPY = "unsupported by the server"


def server_local_tooling_advisory(missing_components: int) -> str:
    """The quiet note that replaces the tooling wall in server mode.

    (task-14827 AC#3) A missing local extra is a fact about a machine
    that is not doing the work, so it is stated as a note rather than
    shown as a ⚠ blocker with an install button beside it -- while still
    being stated, because the same selection imported locally WOULD hit
    it.

    Args:
        missing_components: How many tooling warnings the pre-flight
            raised (the count the wall would have rendered).

    Returns:
        One sentence, no ⚠ glyph.
    """
    noun = "component" if missing_components == 1 else "components"
    verb = "isn't" if missing_components == 1 else "aren't"
    return (
        f"{missing_components} local {noun} {verb} installed — that affects "
        "imports on this machine only; this one runs on the server."
    )


#: (task-14911) The Start gate's opening when the TARGETED backend is the
#: server and it will take nothing in the selection. Deliberately not the
#: local gate's "Nothing in this selection can be imported": these files
#: may well import on this machine (an image does), so the sentence names
#: what is actually true -- they are not going to be *sent*.
INGEST_SERVER_NOTHING_SENDABLE_PREFIX = (
    "Nothing in this selection can be sent to the server"
)

#: (task-14911) ...and the way out. Two of them, neither promised: the
#: switch this canvas already offers, and the set of types the server
#: accepts -- which IS knowable here (``SERVER_ACCEPTED_MEDIA_TYPES`` was
#: established against a live server), unlike its tooling.
INGEST_SERVER_NOTHING_SENDABLE_RECOVERY = (
    "Switch to importing on this machine, or choose video, audio, "
    "document, PDF or e-book files."
)


def server_nothing_sendable_line(forecast: IngestForecast | None) -> str:
    """Render the Start gate's reason for a wholly server-refused selection.

    (task-14911) Read FROM the forecast, never re-derived: task-14823's
    gate asked whether the pre-flight had found a supported type group,
    which is a LOCAL verdict, so a folder of nothing but images forecast
    "0 will be sent to the server · 3 will fail (unsupported by the
    server)" with Start still enabled. The gate and the commit line now
    state the same numbers because there is only one of them.

    Args:
        forecast: The selection's forecast (``targets_server``).

    Returns:
        The gate sentence, naming each blocker by kind. The recovery
        clause is appended only when the server's refusal is one of the
        blockers -- switching target does nothing for a 0-byte file,
        which is refused on both.
    """
    if forecast is None:
        return ""
    parts: list[str] = []
    refused = forecast.will_fail_refused
    empty = forecast.will_fail_empty
    if refused:
        noun = "file" if refused == 1 else "files"
        parts.append(f"{refused} {noun} {INGEST_SERVER_REFUSED_COPY}")
    if empty:
        noun = "file" if empty == 1 else "files"
        parts.append(f"{empty} empty {noun}")
    if not parts:
        # Defensive: the caller only asks when the forecast sends nothing,
        # and every staged file is then refused or empty. Say the count
        # rather than an empty reason.
        staged = forecast.staged_total
        noun = "file" if staged == 1 else "files"
        parts.append(f"{staged} {noun} the server won't take")
    line = f"{INGEST_SERVER_NOTHING_SENDABLE_PREFIX} — {' and '.join(parts)}."
    if refused:
        line += f" {INGEST_SERVER_NOTHING_SENDABLE_RECOVERY}"
    return line


def forecast_summary_line(forecast: IngestForecast | None) -> str:
    """Render the commit-point forecast line from :func:`build_ingest_forecast`.

    Args:
        forecast: The selection's forecast, or ``None``.

    Returns:
        ``"1 will import · 1 will skip · 3 will fail (2 need tooling, 1
        empty)"``-shaped copy; ``""`` when there is no forecast. The
        failure segment names its reasons whenever tooling is one of them
        -- "3 will fail" alone cannot tell a user that installing
        something would change the number.

        Two hedges are carried through rather than rounded off:

        * A SERVER run states what will be *sent* and admits the server's
          tooling was not checked (:data:`INGEST_SERVER_TOOLING_UNKNOWN_COPY`).
        * A capped duplicate probe makes ``will_match`` a floor, which
          makes ``will_import`` (its complement) a CEILING -- stating it
          exactly beside "at least N will match" was arithmetic the user
          could catch out (xhigh review round).
    """
    if forecast is None:
        return ""
    hedged_import = bool(
        forecast.match_capped and forecast.will_match and forecast.will_import
    )
    if forecast.targets_server:
        lead = f"{forecast.will_import} will be sent to the server"
    else:
        lead = f"{forecast.will_import} will import"
    if hedged_import:
        lead = f"at most {lead}"
    parts: list[str] = [lead]
    if forecast.will_match:
        match_text = (
            f"at least {forecast.will_match}"
            if forecast.match_capped
            else str(forecast.will_match)
        )
        parts.append(f"{match_text} will match")
    if forecast.will_skip:
        parts.append(f"{forecast.will_skip} will skip")
    if forecast.will_fail:
        segment = f"{forecast.will_fail} will fail"
        # Reasons a user can ACT on are named even when they are the only
        # one; "N empty" is named only alongside another, because the
        # empty-files line already names those files by name.
        actionable = [
            (count, text)
            for count, text in (
                (forecast.will_fail_refused, INGEST_SERVER_REFUSED_COPY),
                (forecast.will_fail_tooling, "need tooling"),
            )
            if count
        ]
        if actionable:
            reason_count = len(actionable) + bool(forecast.will_fail_empty)
            if reason_count == 1:
                segment += f" ({actionable[0][1]})"
            else:
                fragments = [f"{count} {text}" for count, text in actionable]
                if forecast.will_fail_empty:
                    fragments.append(f"{forecast.will_fail_empty} empty")
                segment += f" ({', '.join(fragments)})"
        parts.append(segment)
    if forecast.targets_server and forecast.will_import:
        parts.append(INGEST_SERVER_TOOLING_UNKNOWN_COPY)
    return " · ".join(parts)


#: (task-3314) The inline two-press consent's fixed opening.
INGEST_START_CONFIRM_PREFIX = "⚠ Press Start again to import anyway"


def forecast_consent_line(forecast: IngestForecast | None) -> str:
    """Render the inline consent line's blast radius FROM the forecast.

    (task-14820 AC#1) Derived, never recomputed: the number here is the
    same field the commit line's failure segment reports, so the two can
    only ever move together.

    Args:
        forecast: The selection's forecast, or ``None``.

    Returns:
        The armed gate line. Files whose group cannot run at all are
        stated as certain ("will fail without more tooling"); files whose
        group is merely degraded keep the softer "may fail".
    """
    doomed = forecast.will_fail_tooling if forecast is not None else 0
    degraded = forecast.at_risk if forecast is not None else 0
    clauses: list[str] = []
    if doomed:
        noun = "file" if doomed == 1 else "files"
        clauses.append(f"{doomed} {noun} will fail without more tooling")
    if degraded:
        noun = "file" if degraded == 1 else "files"
        more = " more" if doomed else ""
        clauses.append(f"{degraded}{more} {noun} may fail")
    if not clauses:
        # Defensive: warnings whose features no staged group claims.
        return f"{INGEST_START_CONFIRM_PREFIX}."
    return f"{INGEST_START_CONFIRM_PREFIX} — {', '.join(clauses)}."


def count_warning_affected_files(preflight: PreflightResult | None) -> int:
    """Distinct staged files whose type group depends on a warned feature.

    (task-3314) The inline consent line's blast radius — the successor to
    the retired guardrail modal's per-feature ``_affected_counts``. A file
    lives in exactly one type group, so summing the affected groups' file
    counts is a distinct-file count.

    (task-14820) Now a thin read of :func:`build_ingest_forecast` rather
    than a second, independent computation: this function BEING that
    second computation is what let the consent line and the commit line
    disagree on screen.

    Args:
        preflight: The active pre-flight result, or ``None``.

    Returns:
        The number of staged (supported-group) files whose group's
        required or optional features include any warned feature. ``0``
        when there is no result, no warnings, or no feature-resolvable
        warnings.
    """
    forecast = build_ingest_forecast(preflight)
    return forecast.consent_affected if forecast is not None else 0


@dataclass(frozen=True)
class LibraryIngestLastSubmission:
    """Snapshot of the last submitted ingest batch (task-3313).

    Captured by the screen at submit time, BEFORE the form auto-clears, so
    "Retry this batch" can re-stage the exact same source with its options
    and metadata. Session-scoped by design (recorded in task-3313's notes):
    the jobs DB persists sources but not the form's staged options, so a
    restart starts with no snapshot and the affordance simply stays hidden.

    Attributes:
        source: The RESOLVED source path/URL that was submitted (not the
            raw typed text — restoring the canonical form is deliberate).
        title: The title field's raw text at submit time.
        author: The author field's raw text at submit time.
        keywords: The keywords field's raw comma-separated text.
        analyze: The "Analyze after import" toggle at submit time.
        chunk: The "Chunk content" toggle at submit time.
        chunk_size: The chunk-size field's raw display text.
        type_options: A per-group COPY of the form's option values.
    """

    source: str
    title: str = ""
    author: str = ""
    keywords: str = ""
    analyze: bool = False
    chunk: bool = False
    chunk_size: str = str(DEFAULT_CHUNK_SIZE)
    type_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass
class LibraryIngestFormState:
    """Mutable form echo for the ingest canvas.

    Owned by the screen as a single bundled field (``self._library_ingest_form``)
    rather than a scatter of scalar attributes, and reset wholesale to
    defaults on rail re-entry into Ingest (see
    ``_reset_library_ingest_transient_state``). Every field here is display
    text only -- validated/coerced values (a resolved path, an int chunk
    size, a keywords tuple) are derived at submit time, never stored back
    into this echo.

    Attributes:
        path: The local file path as typed/picked, unvalidated.
        title: Optional title form field, as typed.
        author: Optional author form field, as typed.
        keywords: Comma-separated keywords, as typed (not yet split).
        analyze: Whether "Analyze after import" is toggled on.
        chunk: Whether "Chunk content" is toggled on.
        chunk_size: The chunk-size field's raw text (display-echo only;
            parsed and clamped to ``[MIN_CHUNK_SIZE, MAX_CHUNK_SIZE]`` at
            submit time, never here).
        advanced_open: Whether the "Advanced options" ``Collapsible`` is
            currently expanded. Synced from the live widget's ``collapsed``
            reactive by the screen's ``Collapsible.Toggled`` handler (both
            a manual click and any future programmatic assignment), and
            read back on every render (``collapsed=not advanced_open``) so
            a recompose -- the analyze/chunk toggle handlers' own, or a
            registry-listener-driven one -- never snaps an expanded panel
            shut out from under the user (mirrors
            ``_rag_search_state.history_collapsed``/
            ``sync_library_rag_history_collapsed`` in ``library_screen.py``).
        expanded_type_groups: Set of type-group ids whose collapsible option
            panels are currently expanded, so user toggles survive
            recomposes.
        type_options: Last-used ingestion options per type group,
            keyed by group id (``pdf``, ``audio_video``, ``ebook``,
            ``generic``).
        preflight: The most recent pre-flight analysis result, if any.
        preflight_checking: Whether a pre-flight analysis is currently
            running (used to show a spinner/disable controls).
    """

    path: str = ""
    title: str = ""
    author: str = ""
    keywords: str = ""
    analyze: bool = field(
        default_factory=lambda: bool(_generic_default("analyze", False))
    )
    chunk: bool = field(default_factory=lambda: bool(_generic_default("chunk", False)))
    chunk_size: str = field(
        default_factory=lambda: str(_generic_default("chunk_size", DEFAULT_CHUNK_SIZE))
    )
    advanced_open: bool = False
    expanded_type_groups: set[str] = field(default_factory=set)
    #: (review round) Whether the "What's missing" tooling fold is open.
    #: Lives here, beside ``expanded_type_groups``, because the canvas
    #: widget is rebuilt by a full recompose and would otherwise reopen
    #: collapsed -- the same reason panel expansion is persisted.
    tooling_detail_expanded: bool = False
    type_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    preflight: PreflightResult | None = None
    preflight_checking: bool = False


@dataclass(frozen=True)
class IngestQueueRow:
    """One rendered row in the ingest canvas's job queue.

    Attributes:
        job_id: The registry-assigned job id (``"ingest-job-{n}"``).
        glyph: The row's leading state glyph -- ``"●"`` for
            queued/parsing/writing, ``"✓"`` for done, ``"✗"`` for failed.
        line: The full rendered row text (binding formats -- see
            ``build_library_ingest_state``). Raw, unescaped: the widget
            layer is responsible for markup-escaping this before it reaches
            a rendered label (a source filename can contain Rich markup
            syntax like ``[/bracket]``).
        can_open: True only for a ``done`` job with a resolved ``media_id``
            -- gates the row's "Open in Library" action.
        can_retry: True only for a ``failed`` job whose ``permanent`` flag
            is ``False`` (M4, fix batch F1b) -- gates the row's "Retry"
            action. A ``permanent`` failure (an unsupported file type or a
            missing source file) will fail the exact same way every time,
            so Retry is withheld entirely rather than offering dead bait;
            ``can_dismiss`` stays available either way.
        can_dismiss: True only for a ``failed`` job -- gates the row's
            "Dismiss" action (L3b AB wave, B2). Currently identical to
            ``can_retry`` (both actions are FAILED-only per the registry's
            ``dismiss``/``requeue`` contracts) but kept as its own field
            rather than reusing ``can_retry`` so the two actions stay
            independently testable if a future change ever lets one apply
            where the other doesn't.
        media_id: The job's resulting media id, when known (``done`` jobs
            only).
    """

    job_id: str
    glyph: str
    line: str
    can_open: bool
    can_retry: bool
    can_dismiss: bool = False
    media_id: int | None = None
    state: IngestJobState | None = None
    #: Where the job runs, mirrored from ``LibraryIngestJob.origin``, so the
    #: widget layer can style or filter by backend without reaching past
    #: this state object into the registry.
    origin: str = "local"
    #: Whether this row offers Cancel. Server batches and active local STT
    #: executor attempts have distinct handlers behind the same affordance.
    can_cancel: bool = False
    #: Whether a local STT row whose cooperative cancel is pending offers the
    #: stronger process-tree stop. Never offered for server work.
    can_force_stop: bool = False
    #: Whether this row offers "View on server". Server-only and done-only, and
    #: additionally requires an id: the server does not always report one, and
    #: an action that cannot resolve anything is worse than no action.
    #: Distinct from ``can_open``, which resolves a LOCAL media row -- a server
    #: ingest has none, so the two are never both true (task-700).
    can_open_on_server: bool = False
    #: The id of the media row the SERVER created, mirrored from the job. Kept
    #: apart from ``media_id`` for the same reason it is on the job: the two id
    #: spaces are unrelated.
    remote_media_id: str | None = None
    source_path: str = ""
    progress: dict[str, Any] | None = None
    error_detail: dict[str, Any] | None = None
    #: True when the job is governed by a durable Research source receipt.
    research_owned: bool = False
    #: (task-2043) Inline error-detail expansion (replaces the old details
    #: toast): whether this row's details are open, and the lines to show.
    details_expanded: bool = False
    detail_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class LibraryIngestCanvasState:
    """Full display state for the Library ingest canvas.

    Attributes:
        header: The canvas header line (always ``"Import media"``).
        server_quiet_line: A muted informational line (``"ingest runs on
            Local"``) shown only when the Library's active runtime source
            is ``"server"`` -- ingest always targets the local media store
            regardless of the browsing scope. Empty when not shown.
        unavailable_line: A blocked-state line explaining why Start is
            disabled, or empty when neither gate is tripped. Precedence:
            a missing registry seam (``registry_available=False``, the
            app-level ingest queue itself is absent) always wins over a
            missing media DB (``media_db_available=False``) -- rendering
            both would be redundant, since without a registry the media-db
            gate can never even be checked in production.
        form: The form echo (see ``LibraryIngestFormState``).
        start_enabled: Whether the "Start import" button is enabled --
            requires a working registry, an available media DB, and a
            non-blank typed path.
        start_quiet_line: (L3b AB wave, A4) A muted line (``"Enter a file
            path to start."``) rendered adjacent to the Start button when
            the path field is blank but both seams are otherwise available.
            Empty once a path is typed, or whenever ``unavailable_line`` is
            already showing -- the db-unavailable/ingest-unavailable lines
            always take precedence so at most one gate line ever renders.
        queue_heading: The queue section heading (always ``"Queue"``).
        queue_counts_line: A per-state job counts summary (L3b AB wave, A2;
            F3 re-anchor) -- empty when the queue itself is empty
            (``QUEUE_EMPTY_COPY`` covers that case instead); otherwise only
            non-zero states, ``parsing -> writing -> queued -> done ->
            failed`` order (the in-flight/"hot" stages first, per the F3
            design spec's UI-impact example).
        queue_rows: Newest-first queue rows (mirrors the registry's own
            ``jobs()`` snapshot order -- callers pass that tuple straight
            through, unsorted).
        queue_show_clear_finished: (L3b AB wave, B2) Whether the "Clear
            finished" button should render below the queue rows -- true
            whenever at least one ``done`` or ``failed`` job is present in
            ``jobs`` (computed from the raw jobs, not from ``queue_rows``,
            so a defensively-malformed done-without-``media_id`` row --
            which renders with ``can_open=False`` -- still counts).
        errors: Pre-flight error messages (e.g. path not found, URL
            unreachable) that should render inline in the summary area.
        type_breakdown_line: Human-readable file/type summary built from the
            pre-flight result, e.g. ``"2 PDF documents, 1 plain text file"``.
            Empty when no pre-flight result is available.
        estimate_line: Lightweight estimate of file count and total size,
            e.g. ``"5 files · 1.2 MB"``. Empty when no pre-flight result is
            available.
        warning_lines: Human-readable tooling/guardrail warnings derived from
            the pre-flight result.
        preflight_checking: Whether a pre-flight analysis is currently running.
        expanded_type_groups: Set of type-group ids whose collapsible option
            panels are expanded, copied from the form state so toggles survive
            recomposes.
        type_groups: Ordered list of supported type-group ids from the latest
            pre-flight result (``unsupported`` is excluded -- it lives in
            ``unsupported_files``).
        unsupported_files: Paths from the pre-flight result's ``unsupported``
            group, rendered separately from supported type groups.
        recent_jobs: The most recent terminal jobs (``DONE`` or ``FAILED``)
            from the registry snapshot, limited to 10. Dismissed jobs are
            intentionally excluded because the registry's ``jobs()`` snapshot
            already filters them out.
        transcribe_cpp_configured: Whether a dedicated local GGUF path exists.
            The path itself never enters this render state.
    """

    header: str
    server_quiet_line: str
    unavailable_line: str
    form: LibraryIngestFormState
    start_enabled: bool
    start_quiet_line: str
    commit_summary_line: str
    #: (task-14820/14822) The selection's ONE forecast, carried onto the
    #: state so the canvas's folded tooling summary reads the same object
    #: the commit and consent lines render from. Without it that fold has
    #: no honest file count and must fall back to counting *components*
    #: instead of *files* -- a second count is exactly the defect this
    #: forecast exists to remove. ``None`` before any analysis.
    forecast: IngestForecast | None
    option_errors: tuple[tuple[str, str, str], ...]
    queue_heading: str
    queue_counts_line: str
    queue_rows: tuple[IngestQueueRow, ...]
    queue_show_clear_finished: bool
    #: (task-2015) Two-press confirm: the armed label names what a second
    #: press will destroy; the resting label is plain "Clear finished".
    queue_clear_finished_label: str
    errors: list[str]
    #: ``True`` when the errors are about the path itself, so the canvas
    #: offers a way to pick a different one instead of a Retry that would
    #: fail identically.
    errors_are_path_problem: bool
    #: Orientation copy for a first visit; empty once the user has
    #: typed a path or an analysis has produced a summary.
    intro_lines: tuple[str, ...]
    #: Whether to offer a one-press way to empty the path field.
    show_clear_path: bool
    type_breakdown_line: str
    estimate_line: str
    #: (task-2043) Pre-flight duplicate forecast: "N file(s) appear to
    #: already be in your Library…", empty when none were detected (or the
    #: staged types can't be checked pre-parse).
    duplicate_line: str
    #: (task-2100) The unsupported-files forecast, NAMED (first 3
    #: basenames) and gate-aware: when the whole selection is blocked
    #: the gate line carries the policy and this line only names the
    #: files.
    unsupported_line: str
    empty_line: str
    warning_lines: list[str]
    #: (review round) Pre-flight notes that name no missing feature (the
    #: URL probe's "could not check the link"). Kept apart from
    #: ``warning_lines`` so they are never counted as missing components
    #: and never folded away -- an advisory the user cannot act on by
    #: installing anything belongs in view, not behind "What's missing".
    advisory_lines: tuple[str, ...]
    #: (review round) Survives the canvas recompose that a registry tick
    #: triggers, so an expanded "What's missing" fold does not snap shut
    #: mid-read; the canvas posts ``ToolingDetailToggled`` and the screen
    #: stores it here, the same way option-panel expansion persists.
    tooling_detail_expanded: bool
    preflight_checking: bool
    expanded_type_groups: set[str]
    type_groups: list[str]
    #: (task-2016) Per-supported-group staged-file counts from the active
    #: pre-flight; the canvas words each panel's scope line from this
    #: ("Applies to all X in this import." vs the zero-file phrasing).
    type_group_file_counts: dict[str, int]
    unsupported_files: list[str]
    recent_jobs: list[LibraryIngestJob]
    queue_empty_line: str
    queue_groups: tuple["IngestQueueGroup", ...]
    latest_batch_line: str
    #: Which backend a new ingest will target, so the canvas can say so.
    ingest_backend: str = "local"
    #: Whether to offer switching backends -- only meaningful when a
    #: server is actually configured.
    show_backend_switch: bool = False
    transcribe_cpp_configured: bool = False
    #: (task-3301) Rendered beside the Start gate when "Analyze after
    #: import" is ON but the configured analysis provider cannot actually
    #: be called (no provider configured / missing key). Informational
    #: only -- analysis is optional, so it never disables Start; the same
    #: resolution stamps the job's "analysis skipped" reason, so the
    #: promise made here and the record left on the done row agree.
    analysis_hint_line: str = ""
    #: (task-3304, MI-17) The distinct install commands behind
    #: ``warning_lines``, in first-appearance order -- the summary renders
    #: one compact "Copy install command" button per entry so the command
    #: is recoverable AT the warning, not only inside the guardrail modal.
    warning_commands: tuple[str, ...] = ()
    #: (task-3314) Whether the gate line is currently the two-press Start
    #: confirm ("⚠ Press Start again to import anyway — N files may
    #: fail."). Only ever ``True`` when the screen's armed flag holds AND
    #: the gate is open AND tooling warnings are active, so a stale armed
    #: flag can never paint consent copy the forecast doesn't justify. The
    #: canvas/gate updater key the warning treatment (the
    #: ``-ingest-start-confirm`` class) off this flag.
    start_confirm_armed: bool = False
    #: (task-3313) Whether the "Retry this batch" affordance is visible:
    #: a last-submission snapshot exists AND the queue has settled (no
    #: queued/parsing/writing job). Canvas-level, always-mounted,
    #: display-managed chrome. Derived from
    #: :func:`library_ingest_retry_available`, the same predicate
    #: ``LibraryScreen.check_action`` gates the ``r`` binding on.
    show_retry_last: bool = False
    #: (xhigh review + live-verify round) Whether a destructive re-stage
    #: is awaiting its second press. Drives the affordance's LABEL (see
    #: :func:`library_ingest_retry_label`), which is the only surface the
    #: ``r`` accelerator has to announce a pending consent on.
    retry_confirm_armed: bool = False
    #: (task-14823) Whether the pre-flight has established that NOTHING in
    #: this selection can be imported -- an empty folder, or a folder whose
    #: files are all unsupported/0-byte.
    #: (task-14911) ...by the backend the run is AIMED at: in server mode
    #: it is also True for a selection the server refuses entirely (a
    #: folder of images), which this machine would import happily. The
    #: gate line says which of the two it is. Closes the Start gate above, and
    #: the screen refuses a submit on it directly so no entry point can
    #: manufacture the failure receipt the pre-flight already ruled out.
    #: Distinct from ``not start_enabled``, which is also False for
    #: transient/environmental blockers (blank path, missing media DB)
    #: where a submit is merely premature, not doomed.
    selection_has_nothing_importable: bool = False


def _basename(source_path: str) -> str:
    """Return a path's display basename, falling back to the raw string."""
    return PurePath(source_path).name or source_path


def _format_elapsed(
    started_at: float | None, finished_at: float | None, *, now: float
) -> str:
    """Format a done job's run time as ``"Ns"`` or ``"Nm Ss"``.

    Args:
        started_at: ``time.monotonic()`` timestamp when the job started
            running, or ``None`` (defensive -- should not happen for a
            ``done`` job).
        finished_at: ``time.monotonic()`` timestamp when the job finished,
            or ``None`` (defensive fallback: ``now`` is used instead so a
            malformed job still renders a sane elapsed value rather than
            crashing).
        now: The caller-supplied "current" monotonic time, used only as
            the ``finished_at`` fallback described above.

    Returns:
        ``""`` when ``started_at`` is unknown (``None`` or the ``0.0``
        default a restored job carries) -- the caller omits the segment;
        ``"<1s"`` under one second; otherwise ``"Ns"`` under a minute, or
        ``"Nm Ss"`` at or above a minute.
    """
    if not started_at:
        # No usable base (None, or the 0.0 default a restored job carries):
        # claiming "0s" would be a lie -- the caller drops the segment.
        return ""
    end = finished_at if finished_at is not None else now
    raw = max(0.0, end - started_at)
    if raw < 1:
        # (task-2015) A watched sub-second job saying "0s" reads as broken.
        return "<1s"
    total_seconds = int(round(raw))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds}s"


_INGEST_PROGRESS_PHASE_LABELS: dict[str, str] = {
    "preparing": "Preparing import",
    "loading": "Loading source",
    "transcribing": "Transcribing audio",
    "post-processing": "Post-processing audio",
    "inspecting": "Inspecting file",
    "extracting": "Extracting",
    "processing": "Processing content",
    "chunking": "Chunking extracted text",
    "analyzing": "Analyzing content",
    "writing": "Saving to Library",
}


def format_ingest_progress_line(
    progress: Mapping[str, Any] | None, *, state: IngestJobState
) -> str:
    """Format one quiet, truthful ingest progress detail line.

    A percentage is shown only when telemetry supplies a finite value within
    its documented bounds. The lifecycle state belongs to the primary queue
    row, so this detail line never repeats it.

    Args:
        progress: Optional structured progress payload.
        state: Current ingest lifecycle state used for the quiet fallback.

    Returns:
        A bounded single-line progress description. Fractional percentages
        are floored so incomplete work never renders as complete.
    """
    payload = progress or {}
    message = payload.get("message")
    text = " ".join(message.split()) if isinstance(message, str) else ""
    if text:
        text = text[:INGEST_PARSE_PROGRESS_MESSAGE_MAX_CHARS]
    if not text:
        phase = payload.get("phase")
        text = _INGEST_PROGRESS_PHASE_LABELS.get(phase, "")
        if not text and state is IngestJobState.PARSING:
            text = "Preparing import"
    percent = payload.get("percent")
    if (
        text
        and isinstance(percent, (int, float))
        and not isinstance(percent, bool)
        and 0 <= percent <= 100
        and math.isfinite(percent)
    ):
        return f"{math.floor(percent)}% · {text}"
    return text


def _build_queue_row_for_state(job: LibraryIngestJob, *, now: float) -> IngestQueueRow:
    """Build one ``IngestQueueRow`` from a registry job snapshot.

    Binding row-line formats (see the L3b plan; F3 splits the old single
    ``running`` row into ``parsing``/``writing``):

    - parsing: ``"● parsing · {basename}"``, plus ``" · {detected_type}"``
      appended when the seam has reported one.
    - writing: ``"● writing · {basename}"``, plus ``" · {detected_type}"``
      appended when the seam has reported one (persisted across the
      ``PARSING`` -> ``WRITING`` transition -- see
      ``LibraryIngestJobRegistry.mark_writing``).
    - queued: ``"● queued · {basename}"``.
    - done: ``"✓ done · {basename} · {elapsed}"`` -- elapsed measured from
      ``submitted_at`` (what the user actually waited, task-2015), the
      `` · {elapsed}`` segment dropped when no usable timestamp exists.
    - failed: ``"✗ failed · {basename} · {short_error}"``, where
      ``short_error`` (L4, fix batch F1b) drops a trailing
      ``" Supported types: ..."`` tail from ``job.error`` so it is not
      repeated on every failed row. An error without that exact marker
      passes through whole. Once ``job.retry_count`` is nonzero
      (task 161), a `` · retry {n}`` suffix is appended.
    """
    basename = _basename(job.source_path)
    if job.state == IngestJobState.PARSING:
        line = f"{_GLYPH_ACTIVE} parsing · {basename}"
        if job.detected_type:
            line += f" · {job.detected_type}"
        # (Qodo round) The attempt marker is the row's LAST element in every
        # state -- appending it before detected_type made it read
        # "… · attempt 2 · plaintext" only on rows that had a type.
        line += _retry_suffix(job)
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_ACTIVE,
            line=line,
            can_open=False,
            can_retry=False,
            media_id=job.media_id,
            state=job.state,
            source_path=job.source_path,
            progress=job.progress,
            error_detail=job.error_detail,
        )
    if job.state == IngestJobState.WRITING:
        line = f"{_GLYPH_ACTIVE} writing · {basename}"
        if job.detected_type:
            line += f" · {job.detected_type}"
        line += _retry_suffix(job)
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_ACTIVE,
            line=line,
            can_open=False,
            can_retry=False,
            media_id=job.media_id,
            state=job.state,
            source_path=job.source_path,
            progress=job.progress,
            error_detail=job.error_detail,
        )
    if job.state == IngestJobState.QUEUED:
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_ACTIVE,
            line=(
                f"{_GLYPH_ACTIVE} queued · {basename}{_retry_suffix(job)}"
            ),
            can_open=False,
            can_retry=False,
            media_id=job.media_id,
            state=job.state,
            source_path=job.source_path,
            progress=job.progress,
            error_detail=job.error_detail,
        )
    if job.state == IngestJobState.DONE:
        # (task-2015) Elapsed is what the user actually waited: submission to
        # finish. ``started_at`` (parse start) excluded the queue wait, so a
        # watched multi-second job could claim "0s".
        elapsed = _format_elapsed(
            job.submitted_at or job.started_at, job.finished_at, now=now
        )
        # (task-2837) The forecast promised "will match"; the receipt says
        # so too. Matched rows carry their own glyph and word, so an import
        # and a dedup match are distinguishable at a glance rather than
        # only by the sub-line.
        is_matched = bool(
            str((job.progress or {}).get("message", "")).startswith(
                INGEST_DUPLICATE_PROGRESS_PREFIX
            )
        )
        glyph = _GLYPH_MATCHED if is_matched else _GLYPH_DONE
        word = "matched" if is_matched else "done"
        line = f"{glyph} {word} · {basename}"
        if elapsed:
            line += f" · {elapsed}"
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=glyph,
            line=line,
            can_open=job.media_id is not None,
            can_retry=False,
            # A server ingest wrote to the server's library, so there is no
            # local row to open; its own action stands in, when the server told
            # us which row it made.
            can_open_on_server=(
                job.origin == "server" and bool(job.remote_media_id)
            ),
            remote_media_id=job.remote_media_id,
            media_id=job.media_id,
            state=job.state,
            source_path=job.source_path,
            progress=job.progress,
            error_detail=job.error_detail,
        )
    # FAILED -- the only remaining IngestJobState member.
    # (task-3305) The detail never repeats the basename the row leads with.
    short_error = _strip_basename_echo(short_ingest_error(job.error), basename)
    if job.state == IngestJobState.CANCELLED:
        # Neither ✓ nor ✗: the user stopped this on purpose, so it is not an
        # error they caused. Retry is withheld because ``requeue`` is
        # FAILED-only and would no-op; dismissing the row is still offered.
        line = f"{_GLYPH_CANCELLED} cancelled · {basename}"
        if job.error:
            line += f" · {short_error}"
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_CANCELLED,
            line=line,
            can_open=False,
            can_retry=False,
            can_dismiss=True,
            media_id=job.media_id,
            state=job.state,
            source_path=job.source_path,
            progress=job.progress,
            error_detail=job.error_detail,
        )

    if job.state == IngestJobState.SKIPPED:
        # (task-2220) Neutral outcome: the pipeline never attempted this
        # file. No Retry (requeue is FAILED-only); dismiss offered.
        line = f"{_GLYPH_SKIPPED} skipped · {basename}"
        if job.error:
            line += f" · {short_error}"
        return IngestQueueRow(
            job_id=job.job_id,
            glyph=_GLYPH_SKIPPED,
            line=line,
            can_open=False,
            can_retry=False,
            can_dismiss=True,
            media_id=job.media_id,
            state=job.state,
            source_path=job.source_path,
            progress=job.progress,
            error_detail=job.error_detail,
        )

    is_unsupported = (
        job.error_detail is not None
        and job.error_detail.get("category") == "unsupported_file_type"
    )
    return IngestQueueRow(
        job_id=job.job_id,
        glyph=_GLYPH_FAILED,
        line=f"{_GLYPH_FAILED} failed · {basename} · {short_error}{_retry_suffix(job)}",
        can_open=False,
        can_retry=not job.permanent and not is_unsupported,
        can_dismiss=True,
        media_id=job.media_id,
        state=job.state,
        source_path=job.source_path,
        progress=job.progress,
        error_detail=job.error_detail,
    )


# (F3 re-anchor) Fixed left-to-right order for the counts line -- the
# in-flight/"hot" pipeline stages first (``parsing``, ``writing``), then the
# backlog (``queued``), then the terminal outcomes (``done``, ``failed``).
# Deliberately its own tuple rather than iterating ``IngestJobState``
# directly: the enum's own declaration order (``QUEUED`` first, matching
# ``LibraryIngestJob.state``'s default) is unrelated to -- and must stay free
# to diverge from -- this display convention.
_COUNTS_LINE_ORDER: tuple[IngestJobState, ...] = (
    IngestJobState.PARSING,
    IngestJobState.WRITING,
    IngestJobState.QUEUED,
    IngestJobState.DONE,
    IngestJobState.SKIPPED,
    IngestJobState.FAILED,
    IngestJobState.CANCELLED,
)


def _queue_counts_line(jobs: Sequence[LibraryIngestJob]) -> str:
    """Build the per-state job counts summary line (L3b AB wave, A2).

    Empty when ``jobs`` is empty (the canvas shows ``QUEUE_EMPTY_COPY``
    instead in that case -- see ``build_library_ingest_state``). Otherwise
    lists only the non-zero ``IngestJobState`` values, always in
    ``_COUNTS_LINE_ORDER`` (``parsing, writing, queued, done, failed``, F3
    re-anchor) so the segment order never shifts as jobs move between
    states -- only which segments are present does. Each segment is
    ``"{n} {state}"`` (no "job"/"jobs" noun, unlike ``count_noun`` elsewhere
    in this module -- e.g. ``"2 parsing · 1 writing · 3 queued · 1 done · 1
    failed"``), joined by ``" · "``.
    """
    # (Qodo round) The tally must bucket the way the group headers and the
    # rows do: counting every DONE job as "done" made the queue line say
    # "2 done" while a header right below it said "1 done · 1 matched".
    counts = {state.value: 0 for state in IngestJobState}
    matched = 0
    for job in jobs:
        if job.state == IngestJobState.DONE and str(
            (job.progress or {}).get("message", "")
        ).startswith(INGEST_DUPLICATE_PROGRESS_PREFIX):
            matched += 1
        else:
            counts[job.state.value] += 1
    segments = [
        f"{counts[state.value]} {state.value}"
        for state in _COUNTS_LINE_ORDER
        if counts[state.value]
    ]
    if matched:
        segments.append(f"{matched} matched")
    joined = " · ".join(segments)
    # (task-2043) The registry restores prior sessions from the jobs DB, so
    # these totals span ALL ingests -- say so, or a fresh batch's outcome
    # blurs into history.
    # (task-2230) The count is QUEUE-scoped -- it drops when the user
    # clears finished rows, while Recent imports keeps the real history.
    # Saying "all ingests" over a number that shrinks was a lie the label
    # itself denied.
    # (task-3305, MI-14) A trailing "— in queue" self-contradicted whenever
    # every listed segment was a terminal outcome (e.g. "2 done · 1 failed
    # — in queue" -- nothing there is actually still queued).
    # (task-2859 item 4) "This queue:" as a leading scope label instead
    # names WHERE the tally is scoped (this queue's lifetime, as opposed to
    # Recent imports' full history) unconditionally, without ever claiming
    # a segment is in an active, not-yet-run state -- this supersedes the
    # narrower any-active-suffix approach task-3305 first landed with.
    return f"This queue: {joined}" if joined else ""


#: Suffix appended to a queue row for a job that runs on the server. Local is
#: the overwhelmingly common case, so it stays unannotated rather than every
#: row carrying a backend tag.
_SERVER_ROW_SUFFIX = " · on server"

#: States a row can no longer be acted on to stop. Mirrors the registry's
#: own terminal set; kept local rather than importing a private name.
_TERMINAL_ROW_STATES = (
    IngestJobState.DONE,
    IngestJobState.FAILED,
    IngestJobState.CANCELLED,
    IngestJobState.SKIPPED,
)


_LOCAL_STT_PROGRESS_PHASES = frozenset(
    {"preparing", "loading", "transcribing", "post-processing"}
)


def ingest_progress_action_signature(job: LibraryIngestJob) -> tuple[bool, bool]:
    """Return local-STT Cancel and Force-stop availability for ``job``."""
    progress = job.progress or {}
    active_local_stt = (
        job.origin == "local"
        and job.state is IngestJobState.PARSING
        and progress.get("phase") in _LOCAL_STT_PROGRESS_PHASES
    )
    cancel_requested = bool(progress.get("cancel_requested"))
    return (
        active_local_stt and not cancel_requested,
        active_local_stt and cancel_requested,
    )


def _build_queue_row(
    job: LibraryIngestJob, *, now: float, details_expanded: bool = False
) -> IngestQueueRow:
    """Build a queue row, then stamp where the job runs.

    The per-state builder returns from one of several branches; annotating the
    origin here rather than in each of them means a new state cannot silently
    ship without the marker. Once local and server ingests share one queue,
    "done · notes.txt" alone cannot tell the user which machine did the work.

    Args:
        job: The job to render.
        now: The "current" monotonic time, for elapsed-time formatting.

    Returns:
        The rendered row, with ``origin`` mirrored and the line marked when the
        job is not local.
    """
    row = _build_queue_row_for_state(job, now=now)
    if job.origin == "local":
        can_cancel, can_force_stop = ingest_progress_action_signature(job)
        row = replace(
            row,
            origin=job.origin,
            can_cancel=can_cancel,
            can_force_stop=can_force_stop,
            research_owned=bool(job.research_source_operation_id),
        )
    else:
        can_cancel = (
            bool(job.batch_id) and job.state not in _TERMINAL_ROW_STATES
        )
        row = replace(
            row,
            origin=job.origin,
            can_cancel=can_cancel,
            line=f"{row.line}{_SERVER_ROW_SUFFIX}",
            research_owned=bool(job.research_source_operation_id),
        )
    if details_expanded and job.error_detail:
        # (task-2043) Inline expansion replaces the old auto-expiring
        # details toast: category + the full (single-prefix) message, plus
        # an honest retry hint when Retry is on offer -- corrupt-file
        # extraction failures stay retryable, but the copy now says what a
        # retry could actually fix.
        # (task-2130) The expansion must never be a verbatim repeat of the
        # row summary: the message line is skipped when it matches the
        # job's own error, the captured exception chain is surfaced, and
        # the retry advisory names a missing dependency instead of
        # gesturing at "missing tooling".
        lines: list[str] = []
        category = str(job.error_detail.get("category") or "").strip()
        if category:
            # (task-2160) No parenthesized exception class -- it serves no
            # user and reads as a leak ("(FileIngestionError)").
            # (task-14821) ...and no raw internal token either: "Category:
            # write error" was shown to users for a failure that never
            # reached a write.
            lines.append(f"Reason: {ingest_failure_reason(category)}")
        message = unwrap_ingest_error(
            str(job.error_detail.get("message") or job.error or "")
        )
        if message and message != unwrap_ingest_error(str(job.error or "")):
            lines.append(f"Details: {message}")
        chain = job.error_detail.get("chain") or ()
        # (task-2140) The worker dedups chain entries against the bare
        # message, but entries carry a "ClassName: " prefix -- strip it
        # before comparing, so an entry that merely restates the row's
        # error (round 5: "Underlying: FileIngestionError: <row error>")
        # never renders.
        # (task-14821) That comparison was exact-equality after ONE
        # ``split(": ", 1)``, which the real ffmpeg failure walked
        # straight through: the message carries a "Failed to ingest audio
        # file: " wrapper the chain entry lacks, so ~40 lines of build
        # flags printed under BOTH Details and Underlying. Containment
        # over the normalized texts survives that drift.
        known_texts = [
            _normalized_detail_text(message),
            _normalized_detail_text(str(job.error or "")),
        ]
        for underlying in tuple(chain)[:3]:
            text_part = _normalized_detail_text(
                str(underlying).split(": ", 1)[-1]
            )
            if _restates_known_text(text_part, known_texts):
                continue
            lines.append(f"Underlying: {underlying}")
        if row.can_retry:
            advice = ingest_retry_advice(
                category=category, message=message, chain=tuple(chain)
            )
            if advice:
                lines.append(advice)
        row = replace(row, details_expanded=True, detail_lines=tuple(lines))
    return row


#: (xhigh review round) One ``Failed to <verb> <type> file:`` stage
#: wrapper, WITHOUT ``_NESTED_FAILURE_PREFIX_RE``'s requirement that
#: another ``Failed to`` follow it. Stripped for COMPARISON only (never
#: from rendered text): the wrapper is added by whichever layer re-raised,
#: so the same failure reads with it in the row message and without it in
#: the chain entry, and a dedup that cannot see past it either prints the
#: payload twice or drops a real root cause.
_STAGE_FAILURE_PREFIX_RE = re.compile(r"^Failed to \w+(?: [\w.+-]+)? file: ")


def _normalized_detail_text(text: str) -> str:
    """Collapse whitespace and stage wrappers for restatement comparison."""
    collapsed = " ".join(unwrap_ingest_error(str(text)).split())
    while True:
        stripped = _STAGE_FAILURE_PREFIX_RE.sub("", collapsed, count=1)
        if stripped == collapsed:
            return collapsed
        collapsed = stripped


def _restates_known_text(candidate: str, known: Sequence[str]) -> bool:
    """Whether ``candidate`` says nothing the already-rendered lines didn't.

    Containment in ONE direction: the candidate is a restatement when it
    is CONTAINED IN something already on screen. A chain entry that
    contains a known text but adds to it is a strict superset -- it quotes
    the row summary and appends the underlying cause -- which is precisely
    the entry the chain exists to surface.

    (xhigh review round) The other direction was tested too, so those
    supersets were discarded: the fix for the duplicated 40-line ffmpeg
    banner (task-14821 AC#4) took the root cause down with it. Wrapper
    drift, the reason equality was not enough there, is handled by
    :func:`_normalized_detail_text` instead.
    """
    if not candidate:
        return True
    return any(text and candidate in text for text in known)


_MISSING_DEPENDENCY_RE = re.compile(
    r"No module named '([^']+)'|(\S+) is not installed|pip install (\S+)"
)

#: (task-14821) Plain-language name for each failure category the ingest
#: pipeline stamps. The expansion used to print the raw token with its
#: underscores swapped ("Category: write error") -- internal vocabulary,
#: and in the no-content case an outright wrong claim: nothing had been
#: written, extraction had produced nothing to write.
_FAILURE_REASON_LABELS: dict[str, str] = {
    "parse_error": "The file couldn't be read.",
    "no_content": "No text could be extracted.",
    "empty_source": "The file is empty.",
    "unsupported_file_type": "This file type isn't supported.",
    "missing_source": "The file couldn't be found.",
    "write_error": "The Library couldn't be written to.",
    "stt_failure": "Transcription failed.",
}

#: (task-14821) Categories whose failure is DETERMINISTIC: the install is
#: missing, or the source carries nothing to extract. Retrying without
#: changing anything reproduces them exactly, so the optimistic advisory
#: must never fire for one.
_DETERMINISTIC_FAILURE_CATEGORIES = frozenset({"no_content", "empty_source"})

#: (task-14821) Phrasings a pipeline error uses when it is really saying
#: "tooling is missing" WITHOUT naming an importable module --
#: ``_MISSING_DEPENDENCY_RE`` cannot match "install an OCR backend
#: (docling, tesseract, easyocr, paddleocr, or docext)", which is exactly
#: the message that was landing in the optimistic branch.
#:
#: (xhigh review round) Every alternative here now has to be a genuine
#: PACKAGING remedy, because the advisory it unlocks tells the user their
#: retry is doomed until they install what the text named. Two were not:
#:
#: * ``is (?:not|un)available`` matched ``TranscriptionError("The shared
#:   local executor is unavailable.")`` -- a pool teardown that clears on
#:   the next attempt -- and answered it with "Retrying now will fail the
#:   same way". Replaced by the ``requested, but X is unavailable`` shape,
#:   which only the deliberate-backend refusals raise
#:   (``Document_Processing_Lib``'s "Docling processing requested, but
#:   Docling is unavailable").
#: * ``may not be installed`` matched the GENERIC extraction refusal
#:   ("...or the tooling for this file type may not be installed"), which
#:   names no tooling at all -- so the advice pointed at a remedy "named
#:   above" that was nowhere on screen.
_TOOLING_REMEDY_RE = re.compile(
    r"install (?:an?|the) [\w\- ]*backend"
    r"|(?:librar(?:y|ies)|dependenc(?:y|ies)|module|package)s? not "
    r"(?:available|installed)"
    r"|requested, but [\w.+-]+ is (?:not |un)available"
    r"|Install (?:it |them )?with:"
    r"|pip install ",
    re.IGNORECASE,
)


def ingest_failure_reason(category: str) -> str:
    """Return the user-readable reason for a failure category.

    Args:
        category: The ``error_detail`` category token.

    Returns:
        A complete sentence. An unmapped token degrades to its own text
        with underscores spaced out rather than raising -- a new category
        must never break the expansion.
    """
    token = str(category or "").strip()
    if not token:
        return ""
    return _FAILURE_REASON_LABELS.get(
        token, f"{token.replace('_', ' ').capitalize()}."
    )


def ingest_retry_advice(
    *, category: str, message: str, chain: Sequence[str] = ()
) -> str:
    """Advice for a retryable failure, derived from its own reason.

    (task-14821) The advisory used to fall through to "A retry can
    succeed if the failure was transient — a busy file or a network
    hiccup" for every category that wasn't ``parse_error`` and every
    message ``_MISSING_DEPENDENCY_RE`` didn't match. A missing-OCR
    failure satisfies both, so the optimistic branch was the COMMON case
    -- printed directly under a row that said an OCR backend was missing,
    and turning Retry into a trap for a deterministic failure.

    (xhigh review round) The tooling sentence claims a remedy was "named
    above". That is only true when the failure text the user is looking at
    actually named one, so the two conditions are separated: a NAMED
    remedy gets the install instruction; a deterministic category with no
    remedy anywhere gets the determinism alone. The generic extraction
    refusal ("...or the tooling for this file type may not be installed")
    is the second case and used to get the first.

    Args:
        category: The failure's ``error_detail`` category.
        message: The failure's (already unwrapped) message.
        chain: Captured underlying exception texts, if any.

    Returns:
        One advisory sentence, or ``""`` when nothing truthful can be
        said -- the unknown case is silent rather than encouraging.
    """
    dependency = _missing_dependency_from(message, tuple(chain))
    if dependency:
        return f"Missing dependency: {dependency}. Install it, then Retry."
    # (xhigh review round) The chain is searched too, and for the same
    # reason ``_missing_dependency_from`` searches it: a real pdf failure
    # on an install without pdf tooling reports ``'NoneType' object has no
    # attribute 'FileDataError'`` as its message and carries the remedy
    # two links down. The chain entries render directly above this line,
    # so "named above" stays true of them.
    if any(
        _TOOLING_REMEDY_RE.search(str(text))
        for text in (message, *tuple(chain))
    ):
        return (
            "Retrying now will fail the same way — install the tooling "
            "named above first, then Retry."
        )
    if category in _DETERMINISTIC_FAILURE_CATEGORIES:
        # Deterministic, but nothing on screen names a remedy -- so state
        # the determinism and stop, rather than sending the user looking
        # for an install instruction that was never given.
        return (
            "Retrying now will fail the same way — this file's content, "
            "or the tooling for it, has to change first."
        )
    if category == "parse_error":
        # (task-2140) No network talk for a local parse failure -- round 5
        # flagged "a network hiccup" advice on a corrupt local PDF as
        # trust-eroding.
        return (
            "If the file is corrupt or truncated, repair or re-export it, "
            "then Retry."
        )
    if category == "write_error":
        # The one cause a retry alone can genuinely clear: the parse
        # succeeded and the Library write did not (a lock, a transient DB
        # error). Nothing about the file itself needs changing.
        return (
            "A retry can succeed if the write failure was temporary — the "
            "file itself parsed fine."
        )
    return ""


@dataclass(frozen=True)
class IngestQueueGroup:
    """One per-submission run of queue rows (task-2221 owner ruling).

    Attributes:
        batch_id: The members' shared batch id, or ``None`` for a
            single-file submission (rendered without a header).
        header_line: Ready-to-render group header, ``""`` for singletons.
        job_ids: The member rows' job ids, in render order.
    """

    batch_id: str | None
    header_line: str
    job_ids: tuple[str, ...]


def _batch_outcome_parts(members: "Sequence[LibraryIngestJob]") -> list[str]:
    """Per-state outcome segments for one batch.

    Args:
        members: The batch's jobs, in render order.

    Returns:
        Non-zero tally segments in ``_COUNTS_LINE_ORDER`` order. Each job
        contributes its actual state; no derived "running" synonym is
        added beside queued/parsing/writing.
    """
    tallies: dict[str, int] = {}
    matched = 0
    for job in members:
        if job.state == IngestJobState.DONE and str(
            (job.progress or {}).get("message", "")
        ).startswith(INGEST_DUPLICATE_PROGRESS_PREFIX):
            # (task-2837) "matched" is reported, not folded into "done".
            matched += 1
        else:
            tallies[job.state.value] = tallies.get(job.state.value, 0) + 1
    parts = [
        f"{tallies[state.value]} {state.value}"
        for state in _COUNTS_LINE_ORDER
        if tallies.get(state.value)
    ]
    if matched:
        parts.append(f"{matched} matched")
    return parts


def build_ingest_queue_groups(
    jobs: "Sequence[LibraryIngestJob]", *, now: datetime | None = None
) -> tuple[tuple[IngestQueueGroup, ...], str]:
    """Group jobs into contiguous per-submission runs (task-2221).

    Contiguous runs of a shared ``batch_id`` become one headed group
    (source dirname, file count, relative age, outcome tallies); jobs
    without a batch id are singleton groups with no header, so a
    single-file submission reads exactly as before. Also returns the
    latest-batch tally line ("Latest batch: …"), ``""`` when no
    multi-file batch exists.

    Args:
        jobs: The registry snapshot, in render order.
        now: Reference time for the header's relative age; defaults to
            the current UTC time.

    Returns:
        ``(groups, latest_batch_line)``.
    """
    reference_now = now if now is not None else datetime.now(timezone.utc)
    groups: list[IngestQueueGroup] = []
    run: list[LibraryIngestJob] = []
    run_batch: str | None = None

    def _flush() -> None:
        if not run:
            return
        members = tuple(run)
        if run_batch is None:
            for member in members:
                groups.append(
                    IngestQueueGroup(
                        batch_id=None, header_line="", job_ids=(member.job_id,)
                    )
                )
            return
        source = PurePath(str(members[0].source_path)).parent.name or "batch"
        count = len(members)
        # (Qodo round) A batch is "running" until EVERY member is
        # terminal -- a finished member's age on an in-progress batch
        # misled about the run's state.
        any_active = any(
            job.state
            in (
                IngestJobState.QUEUED,
                IngestJobState.PARSING,
                IngestJobState.WRITING,
            )
            for job in members
        )
        finished_walls = [
            job.finished_at_wall for job in members if job.finished_at_wall
        ]
        age = (
            format_batch_relative_age(max(finished_walls), now=reference_now)
            if finished_walls and not any_active
            else "active"
        )
        parts = _batch_outcome_parts(members)
        # Whole-branch review M-D (pre-existing conformance fix): no leading
        # "▸ " bullet -- the task-4023 AC#5 glyph convention reserves that
        # prefix for the SELECTED row of a list, and this header is a plain
        # grouping Static (library_ingest_canvas.py), not a row or a
        # disclosure.
        header = (
            f"{source} — {count} {'file' if count == 1 else 'files'}"
            f" · {age}"
        )
        if parts:
            header += " · " + " · ".join(parts)
        groups.append(
            IngestQueueGroup(
                batch_id=run_batch,
                header_line=header,
                job_ids=tuple(job.job_id for job in members),
            )
        )

    for job in jobs:
        job_batch = getattr(job, "batch_id", None)
        if job_batch != run_batch and run:
            _flush()
            run = []
        run_batch = job_batch
        run.append(job)
    _flush()

    # (task-2230) EVERY submission counts as a run, single files included
    # -- filtering to batch_id-bearing groups meant a single-file ingest
    # left this line reporting the previous multi-file batch (round-7 P1,
    # a task-2221 regression). And with only one run in the queue the
    # group header already says it, so the line would just repeat itself.
    latest_batch_line = ""
    if len(groups) > 1:
        by_id = {job.job_id: job for job in jobs}
        latest = max(
            groups,
            key=lambda g: max(
                (by_id[jid].submitted_at for jid in g.job_ids if jid in by_id),
                default=0.0,
            ),
        )
        members = [by_id[jid] for jid in latest.job_ids if jid in by_id]
        parts = _batch_outcome_parts(members)
        if parts:
            latest_batch_line = "Latest run: " + " · ".join(parts)
    return tuple(groups), latest_batch_line


def _missing_dependency_from(message: str, chain: Sequence[str]) -> str:
    """Name the missing dependency when the failure text identifies one.

    (xhigh review round) ``pip install (\\S+)`` is greedy to the next
    space, so a remedy that ends a sentence hands back its full stop:
    the real chain entry ``"...Install with: pip install
    tldw_chatbook[pdf]. Error: No module named 'pymupdf'"`` yielded
    ``tldw_chatbook[pdf].`` and the caller's template added a second dot
    -- ``Missing dependency: tldw_chatbook[pdf]..`` on screen. Sentence
    punctuation is never part of a package name.
    """
    for text in (message, *chain):
        match = _MISSING_DEPENDENCY_RE.search(str(text))
        if match:
            return next(g for g in match.groups() if g).rstrip(".,;:")
    return ""

def build_library_ingest_state(
    jobs: Sequence[LibraryIngestJob],
    *,
    form: LibraryIngestFormState,
    runtime_source: str = "local",
    media_db_available: bool = True,
    registry_available: bool = True,
    now: float | None = None,
    preflight: PreflightResult | None = None,
    preflight_checking: bool | None = None,
    ingest_backend: str = "local",
    server_ingest_available: bool = False,
    transcribe_cpp_configured: bool = False,
    recent_ledger: Sequence[LibraryIngestJob] = (),
    clear_finished_armed: bool = False,
    expanded_details: frozenset[str] | set[str] = frozenset(),
    analysis_unready_hint: str = "",
    start_confirm_armed: bool = False,
    start_confirm_line: str = "",
    last_submission_available: bool = False,
    retry_confirm_armed: bool = False,
) -> LibraryIngestCanvasState:
    """Build the ingest canvas's full display state.

    Args:
        jobs: The registry's current job snapshot (any order accepted --
            typically the registry's own newest-first ``jobs()`` tuple,
            passed straight through into ``queue_rows``).
        form: The current form echo.
        runtime_source: The Library's active runtime scope (``"local"`` or
            ``"server"``); only affects ``server_quiet_line``, since local
            ingest always targets the local media store regardless of
            browsing scope.
        media_db_available: Whether the app's media database seam is
            present. ``False`` blocks Start with ``MEDIA_DB_UNAVAILABLE_COPY``
            (unless ``registry_available`` is also ``False`` -- see
            ``LibraryIngestCanvasState.unavailable_line``).
        registry_available: Whether the app-level ingest job registry seam
            itself is present at all. ``False`` blocks Start with
            ``INGEST_UNAVAILABLE_COPY``, overriding the media-db line.
        now: The "current" monotonic time used for elapsed-time defensive
            fallbacks; defaults to ``time.monotonic()``.
        preflight: Optional pre-flight analysis result. When ``None``, the
            builder falls back to ``form.preflight``.
        preflight_checking: Whether a pre-flight analysis is currently in
            progress. When ``None`` (the default), the value is taken from
            ``form.preflight_checking``.
        transcribe_cpp_configured: Whether the dedicated direct-local model
            setting exists. Only the boolean reaches render state.
        start_confirm_armed: Whether the screen has an immutable pending
            consent snapshot for the current request.
        start_confirm_line: Optional active-duplicate/combined consent copy.
            When empty, an armed tooling-only request retains the forecast's
            existing consent sentence.
        retry_confirm_armed: Whether a destructive re-stage is awaiting its
            second press. Rendered as the affordance's label; ignored
            whenever the affordance itself is hidden.

    Returns:
        The canvas's full display state.
    """
    resolved_now = now if now is not None else time.monotonic()
    active_preflight = preflight if preflight is not None else form.preflight
    active_preflight_checking = (
        form.preflight_checking if preflight_checking is None else preflight_checking
    )
    # The line names the *ingest target*, not the browse scope. On a local-only
    # install there is no choice to explain, so it stays silent; a server target
    # is always named, even if the seam has since gone away, so the canvas can
    # never claim local while submit would still try the server.
    # Server ingest is gated by runtime policy, not just by a configured
    # server: ``media.ingestion_jobs.launch.server`` declares
    # ``required_source="server"``, so the service refuses the launch outright
    # while the Library runtime is local -- the same rule that makes the retired
    # ingest window disable its server panels in local mode. Offering the switch
    # regardless produced a job that failed with "requires server mode".
    in_server_mode = str(runtime_source or "local").strip().lower() == "server"
    server_ingest_offerable = bool(server_ingest_available) and in_server_mode
    targets_server = (
        str(ingest_backend or "local").strip().lower() == "server"
        and server_ingest_offerable
    )
    if targets_server:
        server_quiet_line = INGEST_TARGET_SERVER_COPY
    elif server_ingest_available and not in_server_mode:
        # Explain the precondition instead of letting the submit fail later.
        server_quiet_line = INGEST_SERVER_NEEDS_SERVER_MODE_COPY
    elif server_ingest_offerable:
        server_quiet_line = INGEST_TARGET_LOCAL_COPY
    else:
        server_quiet_line = ""
    if not registry_available:
        unavailable_line = INGEST_UNAVAILABLE_COPY
    elif not media_db_available:
        unavailable_line = MEDIA_DB_UNAVAILABLE_COPY
    else:
        unavailable_line = ""
    queue_rows = tuple(
        _build_queue_row(
            job,
            now=resolved_now,
            details_expanded=job.job_id in expanded_details,
        )
        for job in jobs
    )
    queue_groups, latest_batch_line = build_ingest_queue_groups(jobs)
    # (task-2220 Qodo round) SKIPPED counts as finished everywhere, so it
    # must also SHOW the control -- a skips-only queue was unclearble.
    queue_show_clear_finished = any(
        job.state
        in (
            IngestJobState.DONE,
            IngestJobState.FAILED,
            IngestJobState.SKIPPED,
        )
        for job in jobs
    )
    finished_count = sum(
        1
        for job in jobs
        if job.state
        in (
            IngestJobState.DONE,
            IngestJobState.FAILED,
            IngestJobState.SKIPPED,
        )
    )
    failed_count = sum(
        1 for job in jobs if job.state == IngestJobState.FAILED
    )
    # (task-2160) "finished" includes failed rows -- say so at the moment
    # of destruction, or the user clears their failure records unknowingly.
    failed_suffix = (
        f" (incl. {failed_count} failed)" if failed_count else ""
    )
    queue_clear_finished_label = (
        f"Press again to clear {finished_count} finished{failed_suffix}"
        if clear_finished_armed
        else "Clear finished"
    )

    # Pre-flight summary fields. Copy ``type_groups`` so the frozen
    # ``PreflightResult`` is never mutated; pop ``unsupported`` into its own
    # list for separate rendering.
    if active_preflight is not None:
        type_groups = dict(active_preflight.type_groups)
        unsupported_files = list(type_groups.pop("unsupported", []))
        errors = list(active_preflight.errors)
        if errors:
            # (task-2015) A "0 files" estimate or a type breakdown parked
            # under a path error is noise: error states render the error and
            # its recovery affordance only.
            type_breakdown_line = ""
            estimate_line = ""
        else:
            type_breakdown_line = build_type_breakdown_line(type_groups)
            # (task-3305, MI-19) A URL is not a 0-byte file: the probe
            # cannot know its size, so "1 file · 0 B" was a fabrication.
            # The breakdown line above already names what the URL is.
            if getattr(active_preflight, "source_is_url", False):
                estimate_line = ""
            else:
                estimate_line = build_estimate_line(
                    active_preflight.total_files,
                    active_preflight.total_size,
                    active_preflight.truncated,
                )
        # (review round) A pre-flight warning WITHOUT a ``feature`` key is not
        # a missing component -- the URL probe's "Could not check the link"
        # note is the live case. Feeding it to ``build_warning_lines``
        # produced the nonsense "Could not check the link isn't installed --
        # needed for The site answered 403 ...", and the folded summary then
        # counted it toward "N optional components aren't installed". Split
        # here, at the one place that still has the warning dicts: a rendered
        # line cannot be reverse-engineered back into its warning.
        feature_warnings = [
            warning
            for warning in active_preflight.warnings
            if str((warning or {}).get("feature") or "").strip()
        ]
        advisory_warnings = [
            warning
            for warning in active_preflight.warnings
            if not str((warning or {}).get("feature") or "").strip()
        ]
        warning_lines = build_warning_lines(feature_warnings)
        advisory_lines = tuple(
            line
            for line in (
                _advisory_line(warning) for warning in advisory_warnings
            )
            if line
        )
        warning_commands = preflight_install_commands(feature_warnings)
        if targets_server and warning_lines:
            # (task-14827 AC#3) The wall, its ⚠ summary line and its "Copy
            # install command" button all describe THIS machine's
            # inventory -- and during a server-targeted import this
            # machine does no parsing, so running that pip command would
            # change nothing about the run. Post-14820 the fold at least
            # stopped condemning the import ("no staged file needs them"),
            # but a ⚠ block plus an install button is still a wall of
            # blockers about a machine that isn't doing the work. Demoted
            # to ONE advisory line, which renders as a quiet note with no
            # ⚠ glyph -- the glyph is what carries severity here.
            advisory_lines = advisory_lines + (
                server_local_tooling_advisory(len(warning_lines)),
            )
            warning_lines = []
            warning_commands = ()
        already = getattr(active_preflight, "already_in_library", 0) or 0
        already_capped = bool(
            getattr(active_preflight, "already_in_library_capped", False)
        )
        if already and not errors:
            noun = "file" if already == 1 else "files"
            verb = "appears" if already == 1 else "appear"
            outcome = (
                "it will be matched, not re-imported."
                if already == 1
                else "they'll be matched, not re-imported."
            )
            # (task-2130) When the duplicate check hit its candidate cap the
            # count is a floor -- presenting the cap as the total told an
            # 80-duplicate folder "20 files appear to already be…".
            count_text = f"at least {already}" if already_capped else str(already)
            duplicate_line = (
                f"{count_text} {noun} {verb} to already be in your Library — "
                f"{outcome}"
            )
        else:
            duplicate_line = ""
        errors_are_path_problem = bool(
            errors and getattr(active_preflight, "path_invalid", False)
        )
        type_groups_list = list(type_groups.keys())
    else:
        type_groups = {}
        unsupported_files = []
        errors = []
        errors_are_path_problem = False
        type_breakdown_line = ""
        estimate_line = ""
        duplicate_line = ""
        warning_lines = []
        advisory_lines = ()
        warning_commands = ()
        type_groups_list = []

    # Always expose the generic panel so global options (analyze, chunk) are
    # reachable even when no plain-text files are in the selection.
    if "generic" not in type_groups_list:
        type_groups_list.append("generic")

    # (task-2015) Pre-flight just promised every discovered file will be
    # recorded as a failure -- letting Start stay enabled invites a
    # guaranteed-failure submit. ``type_groups`` here is the post-pop dict of
    # SUPPORTED groups only.
    empty_files = tuple(
        getattr(active_preflight, "empty_files", ()) or ()
    )
    # (task-14820) ONE forecast, consumed by the commit line, the inline
    # consent line, and the nothing-importable gate below -- never
    # recomputed per surface. ``None`` under a path error or before any
    # analysis.
    # (xhigh review round) It is told which backend it is forecasting:
    # ``targets_server`` was computed above and never reached it, so the
    # LOCAL tooling inventory was being used to condemn SERVER runs.
    forecast = build_ingest_forecast(
        active_preflight, targets_server=targets_server
    )
    # (task-14823) A folder holding NOTHING was the one selection this
    # gate let through: ``total_files > 0`` excluded it, so Start stayed
    # enabled with an EMPTY gate line and the press manufactured
    # "✗ failed · emptydir · No files to import were found in this
    # folder." -- a failure the pre-flight had already diagnosed as
    # "0 files · 0 will import".
    empty_selection = (
        active_preflight is not None
        and not errors
        and active_preflight.total_files == 0
    )
    # (xhigh review round) Entries the directory scan passed over without
    # collecting: what tells "this folder is empty" from "this folder's
    # entries were all skipped". Both reach ``empty_selection``.
    skipped_entries = int(getattr(active_preflight, "skipped_entries", 0) or 0)
    nothing_importable = (
        active_preflight is not None
        and not errors
        and not type_groups
        and (active_preflight.total_files > 0 or empty_selection)
    )
    # (task-14911) ...and the same gate, asked of the backend this run is
    # actually aimed at. ``nothing_importable`` above is a LOCAL verdict
    # ("did the pre-flight find a supported type group"), so a folder of
    # nothing but images -- which the server has no media type for at all
    # -- forecast "0 will be sent to the server · 3 will fail (unsupported
    # by the server)" while Start stayed ENABLED and every row landed as a
    # permanent failure. Read from the FORECAST rather than re-deriving a
    # second notion of importability: it already knows what this backend
    # accepts, and the gate line and the commit line then cannot disagree.
    # ``will_match`` guards the task-2223 ruling: zero imports plus
    # predicted duplicate matches keeps Start enabled, because the
    # duplicate probe is capped best-effort and never a blocker.
    nothing_sendable = (
        targets_server
        and forecast is not None
        and forecast.staged_total > 0
        and forecast.will_import == 0
        and forecast.will_match == 0
    )
    # (task-2130) Invalid option values gate Start exactly like a bad path:
    # "abc" as a chunk size used to sail into a running job with only a
    # focus-only colored border as the signal.
    option_errors = collect_ingest_option_errors(
        form.type_options, groups=("generic", *type_groups)
    )
    # (task-2230) An unresolvable source gates like every other
    # known-doomed selection: a nonexistent path used to leave Start
    # styled exactly like a valid one, and pressing it produced a
    # transient toast and NO queue record -- the least recovery for the
    # most common user error.
    start_enabled = (
        registry_available
        and media_db_available
        and bool(form.path.strip())
        and not nothing_importable
        and not nothing_sendable
        and not option_errors
        and not errors_are_path_problem
    )
    # (L3b AB wave, A4) At most one gate line ever renders at once: the
    # unavailable line wins, then the guaranteed-failure explanation, then
    # the blank-path nudge.
    if unavailable_line:
        start_quiet_line = ""
    elif empty_selection:
        # (task-14823 AC#2) An empty folder and an all-unsupported folder
        # need different recoveries -- "add files / pick another folder"
        # versus "these formats aren't supported" -- so they get different
        # sentences rather than one shared blocker line.
        # (xhigh review round) ...and a folder whose entries were all
        # SKIPPED is a third case again: it is not empty, and telling its
        # owner it is -- while refusing the submit -- is a dead end. The
        # gate itself is still correct there, because the submit path
        # walks the folder with the very same collector
        # (``collect_directory_files``) and would queue nothing.
        start_quiet_line = (
            ingest_unscannable_selection_copy(skipped_entries)
            if skipped_entries
            else INGEST_EMPTY_SELECTION_COPY
        )
    elif nothing_importable:
        # (task-2160) Name the blockers by KIND: a solo 0-byte file used to
        # read "1 unsupported file" via the total-files fallback.
        blocker_parts: list[str] = []
        if unsupported_files:
            u = len(unsupported_files)
            blocker_parts.append(
                f"{u} unsupported {'file' if u == 1 else 'files'}"
            )
        if empty_files:
            e = len(empty_files)
            blocker_parts.append(
                f"{e} empty {'file' if e == 1 else 'files'}"
            )
        if not blocker_parts:
            total = active_preflight.total_files
            blocker_parts.append(
                f"{total} unsupported {'file' if total == 1 else 'files'}"
            )
        start_quiet_line = (
            f"Nothing in this selection can be imported — "
            f"{' and '.join(blocker_parts)}."
        )
    elif nothing_sendable:
        # (task-14911) Ordered AFTER ``nothing_importable`` on purpose: a
        # file nothing on this machine can read is diagnosed the same way
        # whichever target is selected, and switching to Local would not
        # help. This branch is the other case -- files this machine reads
        # perfectly well that this destination will not take -- and it
        # needs the server's own vocabulary and its own recovery.
        start_quiet_line = server_nothing_sendable_line(forecast)
    elif errors_are_path_problem:
        start_quiet_line = (
            "Can't find that path — check it, or use Browse… to pick a "
            "file or folder."
        )
    elif option_errors:
        start_quiet_line = (
            f"Fix the highlighted options to start: {option_errors[0][2]}"
        )
    elif not form.path.strip():
        start_quiet_line = START_QUIET_LINE_COPY
    else:
        start_quiet_line = ""

    # (task-2130) A one-line commit summary beside Start for a valid
    # selection: the forecast lives at the top of a long form, and the
    # commit point at the bottom -- restate the outcome where the finger
    # is.
    # (task-14820) It renders whenever there IS a forecast -- including
    # while a gate blocks Start. Hiding it (task-3305, MI-16) cost a
    # blocked user the numbers they were reasoning about; MI-16's actual
    # defect was a STALE line, and the gate updater syncs both lines in
    # one pass, so they move together now.
    # (xhigh review round) With ONE exception, which AC#4 never covered:
    # when the runtime has no import path AT ALL (no registry, no media
    # DB) the forecast is not a blocked user's arithmetic, it is a promise
    # nothing can keep -- "1 will import" beside a permanently dead Start.
    # A blocked-but-real selection (bad option value, armed consent) keeps
    # its numbers, which is what AC#4 is about.
    commit_summary_line = (
        "" if unavailable_line else forecast_summary_line(forecast)
    )
    if forecast is not None:
        # (task-2223 ruling) Zero imports + ≥1 predicted match keeps Start
        # ENABLED (the dedup probe is capped best-effort, never a blocker)
        # but consent becomes informed: say what starting will actually do.
        # (task-2837) "Everything here" must be true: only when every
        # importable file is a predicted match and nothing else is staged.
        if (
            start_enabled
            and forecast.will_import == 0
            and forecast.will_match
            and not forecast.will_fail
            and not forecast.will_skip
        ):
            start_quiet_line = (
                "Everything here appears to already be in your Library — "
                "starting will re-check and match, not re-import."
            )

    # (task-3314) Inline two-press consent: while the screen's armed flag
    # holds, the gate is open, and tooling warnings are active, the gate
    # line becomes the explicit confirm naming the blast radius. Applied
    # LAST among the quiet-line writers on purpose — a pending consent is
    # the acute state and outranks the informational lines above (the
    # unavailable/blocked branches can never coincide with it, since they
    # all imply ``start_enabled`` is False). The armed flag is gated here,
    # not trusted: a stale carrier with no active warnings renders the
    # ordinary gate line and reports ``start_confirm_armed=False``.
    # (xhigh review round) The trigger is the FORECAST's blast radius, not
    # the bare presence of warnings: a server run reads the same local
    # tooling warnings and has nothing at stake in them, so keying off
    # ``warning_lines`` painted "Press Start again to import anyway"
    # followed by no reason at all. ``consent_affected`` is the same field
    # the confirm sentence renders, so the gate and its copy cannot
    # disagree about whether there is anything to consent to.
    start_confirm_active = bool(
        start_confirm_armed
        and start_enabled
        and (
            start_confirm_line
            or (
                warning_lines
                and forecast is not None
                and forecast.consent_affected
            )
        )
    )
    if start_confirm_active:
        # (task-14820 AC#1) Rendered FROM the same forecast the commit
        # line above it renders from -- the two numbers cannot disagree
        # because there is only one of them.
        start_quiet_line = (
            start_confirm_line
            if start_confirm_line
            else forecast_consent_line(forecast)
        )

    # (task-3313) "Retry this batch" appears once a last submission exists
    # AND the queue has settled — an active job means that submission has
    # not reached a terminal state yet, and re-staging mid-run invites a
    # duplicate batch.
    show_retry_last = library_ingest_retry_available(
        jobs, last_submission_available=last_submission_available
    )

    # (task-2100) Name the unsupported files -- a count alone forces a
    # submit-and-read-the-rows round trip to learn WHICH files. When the
    # gate has already blocked the whole selection, the gate line carries
    # the policy and this line only names the offenders (the old copy
    # promised "will be recorded as a failure" beside a submit that never
    # runs).
    if unsupported_files and not errors:
        unsupported_count = len(unsupported_files)
        unsupported_names = ", ".join(
            PurePath(str(f)).name for f in unsupported_files[:3]
        )
        if unsupported_count > 3:
            unsupported_names += ", ..."
        if nothing_importable:
            # (task-2130) Say what WOULD work: the supported-formats
            # sentence lives in the intro lines, which are hidden the
            # moment a path is typed -- exactly when this line renders.
            unsupported_line = (
                f"Unsupported: {unsupported_names}. "
                f"{SUPPORTED_FORMATS_COPY}"
            )
        else:
            file_noun = "file" if unsupported_count == 1 else "files"
            # (task-2220 owner ruling) Skipped, not "recorded as
            # failures" -- the LOCAL pipeline never attempts these.
            # (task-14827) ...but the server does not skip: it records a
            # permanent failure row with a reason, so on that backend the
            # promise of a quiet skip is the same forecast-vs-receipt
            # disagreement task-14820 exists to remove.
            outcome = "will fail" if targets_server else "will be skipped"
            unsupported_line = (
                f"{unsupported_count} unsupported {file_noun} {outcome}: "
                f"{unsupported_names}."
            )
    else:
        unsupported_line = ""

    # (task-2160) The forecast names the 0-byte files it is certain will
    # fail, exactly like unsupported ones -- it used to promise
    # "1 will import" for a file it had just measured at 0 B.
    if empty_files and not errors:
        empty_count = len(empty_files)
        empty_names = ", ".join(
            PurePath(str(f)).name for f in empty_files[:3]
        )
        if empty_count > 3:
            empty_names += ", ..."
        noun = "file" if empty_count == 1 else "files"
        verb = "is" if empty_count == 1 else "are"
        empty_line = (
            f"{empty_count} empty {noun} will fail — {empty_names} "
            f"{verb} 0 B."
        )
    else:
        empty_line = ""

    # Orientation is for an untouched form only: once there is a path or a
    # summary to read, it would just be noise above the real content.
    intro_lines: tuple[str, ...] = ()
    if not form.path.strip() and active_preflight is None:
        intro_lines = build_intro_lines()

    # (task-2130) Recent imports is the durable session ledger: jobs the
    # user cleared from the queue live on here (the screen snapshots them
    # into ``recent_ledger`` before the registry removal), so Clear
    # finished no longer erases the only record of a session's failures.
    recent_jobs = [
        job
        for job in jobs
        if job.state
        in (
            IngestJobState.DONE,
            IngestJobState.FAILED,
            IngestJobState.SKIPPED,
        )
    ]
    live_ids = {job.job_id for job in recent_jobs}
    recent_jobs.extend(
        job for job in recent_ledger if job.job_id not in live_ids
    )
    recent_jobs = recent_jobs[:10]
    queue_empty_line = ""
    if not queue_rows:
        queue_empty_line = (
            QUEUE_EMPTY_AFTER_ACTIVITY_COPY
            if recent_jobs
            else QUEUE_EMPTY_COPY
        )

    return LibraryIngestCanvasState(
        header=INGEST_HEADER_COPY,
        server_quiet_line=server_quiet_line,
        ingest_backend="server" if targets_server else "local",
        show_backend_switch=server_ingest_offerable,
        unavailable_line=unavailable_line,
        form=form,
        start_enabled=start_enabled,
        start_quiet_line=start_quiet_line,
        commit_summary_line=commit_summary_line,
        forecast=forecast,
        option_errors=option_errors,
        queue_heading=QUEUE_HEADING_COPY,
        queue_counts_line=_queue_counts_line(jobs),
        queue_rows=queue_rows,
        queue_show_clear_finished=queue_show_clear_finished,
        queue_clear_finished_label=queue_clear_finished_label,
        errors=errors,
        errors_are_path_problem=errors_are_path_problem,
        intro_lines=intro_lines,
        show_clear_path=bool(form.path.strip()),
        type_breakdown_line=type_breakdown_line,
        estimate_line=estimate_line,
        duplicate_line=duplicate_line,
        unsupported_line=unsupported_line,
        empty_line=empty_line,
        warning_lines=warning_lines,
        advisory_lines=advisory_lines,
        tooling_detail_expanded=bool(
            getattr(form, "tooling_detail_expanded", False)
        ),
        preflight_checking=active_preflight_checking,
        expanded_type_groups=set(form.expanded_type_groups),
        type_groups=type_groups_list,
        type_group_file_counts={
            group: len(paths) for group, paths in type_groups.items()
        },
        unsupported_files=unsupported_files,
        recent_jobs=recent_jobs,
        queue_empty_line=queue_empty_line,
        queue_groups=queue_groups,
        latest_batch_line=latest_batch_line,
        transcribe_cpp_configured=transcribe_cpp_configured,
        # (task-3301) Only meaningful while the Analyze toggle is ON; the
        # caller supplies the resolved-unready sentence, this builder
        # gates it on the toggle so an OFF form never nags about a
        # provider it isn't going to use.
        analysis_hint_line=(analysis_unready_hint if form.analyze else ""),
        warning_commands=tuple(warning_commands),
        start_confirm_armed=start_confirm_active,
        show_retry_last=show_retry_last,
        # (xhigh review + live-verify round) Gated on visibility: a
        # stale armed flag can never label a hidden affordance.
        retry_confirm_armed=bool(retry_confirm_armed) and show_retry_last,
        selection_has_nothing_importable=bool(
            nothing_importable or nothing_sendable
        ),
    )


def parse_keywords(raw: str) -> tuple[str, ...]:
    """Comma-split a keywords form field, stripping and dropping empties.

    Args:
        raw: The raw, comma-separated keywords text.

    Returns:
        A tuple of non-empty, stripped keyword strings.
    """
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def clamp_chunk_size(raw: str) -> int:
    """Parse and clamp a chunk-size form field at submit time.

    Args:
        raw: The chunk-size field's raw display text.

    Returns:
        The parsed integer clamped to ``[MIN_CHUNK_SIZE, MAX_CHUNK_SIZE]``,
        or ``DEFAULT_CHUNK_SIZE`` when ``raw`` does not parse as an int.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_SIZE
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, value))
