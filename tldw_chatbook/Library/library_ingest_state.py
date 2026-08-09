"""Pure display-state for the Library ingest canvas.

Renders the app-level Library ingest job registry (``library_ingest_jobs.py``)
plus a small local form echo into the immutable state
``LibraryIngestCanvas`` (the widget in ``Widgets/Library/library_ingest_canvas.py``)
renders from. Textual-free (stdlib only) so it is unit-testable without
booting the TUI, mirroring ``library_notes_sync_state.py``.
"""

from __future__ import annotations

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
from tldw_chatbook.Library.ingest_capabilities import (
    MULTI_PAGE_SCRAPE_METHODS,
    _is_installed as _dependency_installed,
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
    "e-books, plain text files, web pages (by URL)."
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


def validate_ingest_option_value(field: Any, value: Any) -> str:
    """Validation message for one option value, or ``""`` when valid.

    (task-2130) Shared by the state gate and the canvas's inline per-field
    messages so the two can never disagree. Only ``number`` fields have a
    wrong shape today; other types are constrained by their widgets. The
    chunk-size bounds mirror ``clamp_chunk_size``'s submit-time clamp
    (Qodo round: a value the UI blessed must not be silently rewritten at
    submit).

    Args:
        field: The ``OptionField`` schema entry for the value.
        value: The raw form value (display text for Inputs).

    Returns:
        A human-readable problem statement, or ``""`` when the value is
        acceptable to every downstream consumer.
    """
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
        for field in cap.fields:
            if field.depends_on is not None and not _dependency_installed(
                field.depends_on
            ):
                continue
            if field.enabled_when is not None:
                gate = fields_by_name.get(field.enabled_when)
                gate_value = values.get(
                    field.enabled_when,
                    gate.default if gate is not None else False,
                )
                if field.enabled_when_values:
                    if gate_value not in field.enabled_when_values:
                        continue
                elif not bool(gate_value):
                    continue
            message = validate_ingest_option_value(
                field, values.get(field.name, field.default)
            )
            if message:
                errors.append((group, field.name, message))
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


def short_ingest_error(error: str) -> str:
    """Return the short (queue-row) form of an ingest job's error message.

    Drops the trailing ``" Supported types: ..."`` tail that
    ``local_file_ingestion.py``'s "Unsupported file type" error carries --
    that tail is dropped from the queue-row summary so it is not repeated
    on every failure surface. An error without that exact marker passes
    through whole.

    Single source of truth for BOTH failure-reason surfaces: the Library
    ingest queue row (``_build_queue_row``) and Home's failed-item canvas
    line (``active_work_adapter._local_ingest_job_items``) call this same
    helper, so the two can never drift apart (F1b whole-wave review).

    Args:
        error: The raw ``LibraryIngestJob.error`` text.

    Returns:
        The error up to (excluding) the supported-types marker, right-
        stripped; the whole error when the marker is absent.
    """
    return unwrap_ingest_error(error).split(
        _SUPPORTED_TYPES_ERROR_MARKER
    )[0].rstrip()


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
            ``_library_rag_history_collapsed``/
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
        progress = job.progress or {}
        executor_phase = progress.get("phase") in {
            "preparing",
            "loading",
            "transcribing",
            "post-processing",
        }
        cancel_requested = bool(progress.get("cancel_requested"))
        active_local_stt = job.state is IngestJobState.PARSING and executor_phase
        row = replace(
            row,
            origin=job.origin,
            can_cancel=active_local_stt and not cancel_requested,
            can_force_stop=active_local_stt and cancel_requested,
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
            lines.append(f"Category: {category.replace('_', ' ')}")
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
        known_texts = {
            message,
            unwrap_ingest_error(str(job.error or "")),
        }
        for underlying in tuple(chain)[:3]:
            text_part = str(underlying).split(": ", 1)[-1]
            if unwrap_ingest_error(text_part) in known_texts:
                continue
            lines.append(f"Underlying: {underlying}")
        if row.can_retry:
            dependency = _missing_dependency_from(message, tuple(chain))
            if dependency:
                lines.append(
                    f"Missing dependency: {dependency}. Install it, then "
                    "Retry."
                )
            elif category == "parse_error":
                # (task-2140) No network talk for a local parse failure --
                # round 5 flagged "a network hiccup" advice on a corrupt
                # local PDF as trust-eroding.
                lines.append(
                    "If the file is corrupt or truncated, repair or "
                    "re-export it, then Retry."
                )
            else:
                lines.append(
                    "A retry can succeed if the failure was transient — a "
                    "busy file or a network hiccup. If the file itself is "
                    "corrupt, repair or re-export it first."
                )
        row = replace(row, details_expanded=True, detail_lines=tuple(lines))
    return row


_MISSING_DEPENDENCY_RE = re.compile(
    r"No module named '([^']+)'|(\S+) is not installed|pip install (\S+)"
)


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
    """Per-state outcome segments for one batch, active work last.

    Args:
        members: The batch's jobs, in render order.

    Returns:
        Non-zero tally segments in ``_COUNTS_LINE_ORDER`` order (e.g.
        ``["2 done", "1 skipped"]``), with a trailing ``"N running"``
        segment when any member is still queued/parsing/writing.
    """
    tallies: dict[str, int] = {}
    active = 0
    matched = 0
    for job in members:
        if job.state in (
            IngestJobState.QUEUED,
            IngestJobState.PARSING,
            IngestJobState.WRITING,
        ):
            active += 1
        elif job.state == IngestJobState.DONE and str(
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
    if active:
        parts.append(f"{active} running")
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
            else "running"
        )
        parts = _batch_outcome_parts(members)
        header = (
            f"▸ {source} — {count} {'file' if count == 1 else 'files'}"
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
    """Name the missing dependency when the failure text identifies one."""
    for text in (message, *chain):
        match = _MISSING_DEPENDENCY_RE.search(str(text))
        if match:
            return next(g for g in match.groups() if g)
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
        warning_lines = build_warning_lines(active_preflight.warnings)
        warning_commands = preflight_install_commands(active_preflight.warnings)
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
    nothing_importable = (
        active_preflight is not None
        and not errors
        and active_preflight.total_files > 0
        and not type_groups
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
        and not option_errors
        and not errors_are_path_problem
    )
    # (L3b AB wave, A4) At most one gate line ever renders at once: the
    # unavailable line wins, then the guaranteed-failure explanation, then
    # the blank-path nudge.
    if unavailable_line:
        start_quiet_line = ""
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
    # is. Only renders when the gate is open and there is a real forecast.
    commit_summary_line = ""
    if start_enabled and active_preflight is not None and not errors:
        supported_total = sum(
            len(files) for files in type_groups.values()
        )
        will_match = getattr(active_preflight, "already_in_library", 0) or 0
        match_capped = bool(
            getattr(active_preflight, "already_in_library_capped", False)
        )
        will_skip = len(unsupported_files)
        will_fail = len(empty_files)
        will_import = max(supported_total - will_match, 0)
        parts: list[str] = [f"{will_import} will import"]
        if will_match:
            match_text = (
                f"at least {will_match}" if match_capped else str(will_match)
            )
            parts.append(f"{match_text} will match")
        if will_skip:
            parts.append(f"{will_skip} will skip")
        if will_fail:
            parts.append(f"{will_fail} will fail")
        commit_summary_line = " · ".join(parts)
        # (task-2223 ruling) Zero imports + ≥1 predicted match keeps Start
        # ENABLED (the dedup probe is capped best-effort, never a blocker)
        # but consent becomes informed: say what starting will actually do.
        # (task-2837) "Everything here" must be true: only when every
        # importable file is a predicted match and nothing else is staged.
        if (
            will_import == 0
            and will_match
            and not will_fail
            and not will_skip
        ):
            start_quiet_line = (
                "Everything here appears to already be in your Library — "
                "starting will re-check and match, not re-import."
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
            # failures" -- the pipeline never attempts these.
            unsupported_line = (
                f"{unsupported_count} unsupported {file_noun} will be "
                f"skipped: {unsupported_names}."
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
