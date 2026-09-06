"""Library Browse ▸ Media canvas: media list, type filter, and preview."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Audio.meeting_session import (
    MEETING_JSON,
    TRANSCRIPT_JSONL,
    MeetingSegment,
    format_clock,
    is_widget_safe_cluster_id,
    normalize_speaker_name,
    read_meeting_json,
    render_label,
    update_meeting_json,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import ConflictError, InputError, dispatch_media_post_ingest
from tldw_chatbook.Library.library_pager_state import LibraryPagerDisplay
from tldw_chatbook.Library.library_media_state import (
    LibraryMediaCanvasState,
    MEDIA_SORT_CHOICES,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_DELETE_SELECTED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
    LIBRARY_ANALYZE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_ANALYZE_SELECTED_TOOLTIP,
    LIBRARY_REVIEW_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_REVIEW_SELECTED_TOOLTIP,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
    library_choice_label,
    library_choice_tooltip,
    library_disabled_action_label,
)
from tldw_chatbook.Utils.log_sanitizer import redact_user_paths
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


# ---- Task 8 (meeting diarization spec): rename speakers after the meeting --
# A finished meeting is an ordinary Library Media item; its recording folder
# (holding `meeting.json`'s name map and `transcript.jsonl`'s per-segment
# records) is reachable for free as `Media.url`'s parent directory (`url` is
# the `mixed.wav` path). Both survive raw-track cleanup.
_MEETING_USER_DISPLAY_NAME = "You"  # mirrors meetings_screen.py's LABELS["you"]


def can_rename_meeting_speakers(db: Any, media_id: int) -> bool:
    """True only while `media_id`'s meeting folder still holds a `meeting.json`.

    False (never an exception) when the media item is gone, has no URL, or
    the folder has since been cleaned up.

    Reads ONLY the `url` column, never the row: this runs on the UI thread
    for every media selection, filter and page change (`_library_media_
    canvas_presentation`), and `get_media_by_id`'s `SELECT *` dragged the
    whole `content` blob of every non-meeting item across with it (fix I3).
    """
    try:
        cursor = db.execute_query(
            "SELECT url FROM Media WHERE id = ? AND deleted = 0 AND is_trash = 0", (media_id,)
        )
        row = cursor.fetchone()
    except Exception:  # noqa: BLE001 - a lookup failure just means "can't tell, so no"
        return False
    url = row["url"] if row else None
    if not url:
        return False
    return (Path(url).parent / MEETING_JSON).exists()


def _read_meeting_transcript_segments(folder: Path) -> list[MeetingSegment]:
    """Parse the folder's `transcript.jsonl` into `MeetingSegment`s, in order."""
    path = folder / TRANSCRIPT_JSONL
    if not path.exists():
        return []
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            segments.append(MeetingSegment(**json.loads(line)))
    return segments


def _render_meeting_transcript(segments: list[MeetingSegment], names: dict[str, str]) -> str:
    """Render `segments` as the same "[hh:mm:ss] Label: text" lines the live
    meeting screen shows, using the CURRENT name map (task 2's `render_label`)."""
    lines = []
    for segment in segments:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        label = render_label(segment, names, _MEETING_USER_DISPLAY_NAME)
        lines.append(f"{stamp} {label}: {segment.text}" if label else f"{stamp} {segment.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _write_meeting_transcript_row(
    db: Any,
    conn: Any,
    media_id: int,
    whisper_model: str | None,
    transcription: str,
    now: str,
    client_id: str,
) -> None:
    """Insert or version-bump this media's `Transcripts` row (same connection
    and transaction as the `Media` rewrite below).

    No public "add/update a Transcripts row" method exists anywhere on
    `MediaDatabase` -- ingestion never writes this table in this codebase
    today (only reads/deletes are exposed) -- so this mirrors the sync
    contract every other write in `Client_Media_DB_v2.py` follows by hand:
    an optimistic-locked, version-incrementing UPDATE (or a fresh INSERT
    when there's no prior row), each followed by a `_log_sync_event` on the
    SAME connection/transaction as the `Media` write.
    """
    cur = conn.cursor()
    whisper_model = whisper_model or "meeting"
    # Match regardless of `deleted` (final whole-branch review M2): the table's
    # UNIQUE(media_id, whisper_model) counts soft-deleted rows too, so a prior
    # soft-deleted row that a `deleted = 0` filter skips would collide the
    # INSERT below (a caught IntegrityError that silently dropped the rename).
    # Reuse the row instead -- the UPDATE path resurrects it (deleted=0).
    # ponytail: keys on `media_id` alone, not `(media_id, whisper_model)` --
    # fine today because no ASR pipeline in this codebase writes a Transcripts
    # row for a meeting item, so at most one row per media_id ever exists here.
    cur.execute(
        "SELECT id, uuid, version, deleted FROM Transcripts WHERE media_id = ? ORDER BY id DESC LIMIT 1",
        (media_id,),
    )
    existing = cur.fetchone()
    if existing is None:
        transcript_uuid = db._generate_uuid()
        cur.execute(
            "INSERT INTO Transcripts (media_id, whisper_model, transcription, created_at, uuid, "
            "last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?, 1, ?, 0)",
            (media_id, whisper_model, transcription, now, transcript_uuid, now, client_id),
        )
        db._log_sync_event(
            conn, "Transcripts", transcript_uuid, "create", 1,
            {
                "media_id": media_id, "whisper_model": whisper_model, "transcription": transcription,
                "last_modified": now, "version": 1, "client_id": client_id,
            },
        )
    else:
        new_version = existing["version"] + 1
        # deleted=0 resurrects a soft-deleted prior row (M2) -- a no-op when the
        # row was already live.
        cur.execute(
            "UPDATE Transcripts SET transcription=?, last_modified=?, version=?, client_id=?, prev_version=?, deleted=0 "
            "WHERE id=? AND version=?",
            (transcription, now, new_version, client_id, existing["version"], existing["id"], existing["version"]),
        )
        if cur.rowcount == 0:
            raise ConflictError(f"Transcripts rename conflict for id={existing['id']}", existing["id"])
        db._log_sync_event(
            conn, "Transcripts", existing["uuid"], "update", new_version,
            {"transcription": transcription, "last_modified": now, "version": new_version, "client_id": client_id},
        )


@dataclass(frozen=True)
class SpeakerRenameResult:
    """Outcome of `rename_meeting_speaker`: did it write, and if not, why.

    `reason` is static, user-safe copy (never a path, a name, or transcript
    text) so the canvas can show it verbatim.
    """

    ok: bool
    reason: str = ""


#: A refusal leaves BOTH authorities untouched (fix C2 / Qodo Q16).
RENAME_REFUSED_NOT_MEETING_CONTENT = (
    "this item's Library transcript wasn't produced by the meeting recording"
)
RENAME_REFUSED_EMPTY_TRANSCRIPT = "this meeting's local transcript is missing or empty"


def rename_meeting_speaker(db: Any, media_id: int, cluster_id: str, name: str) -> SpeakerRenameResult:
    """Rename `cluster_id` to `name` on a finished meeting's Library item.

    Updates the meeting folder's `meeting.json` name map (an empty `name`
    removes the entry), re-renders every `transcript.jsonl` segment's text
    via `render_label`, and rewrites `Media.content` -- the field the media
    canvas displays and `media_fts` indexes -- in ONE DB transaction that
    also writes a new versioned `Transcripts` row.

    `add_media_with_keywords` (the usual "update an existing Media row"
    entrypoint, `Client_Media_DB_v2.py` Case A.1.b) is deliberately NOT
    reused here: nested inside this function's own transaction it would
    fire the post-ingest hook (RAG re-indexing) before the outer commit
    lands, and its full-row payload would need every untouched column
    (author, ingestion_date, transcription provenance...) round-tripped
    just to avoid clobbering them with defaults. Scoping the UPDATE to only
    `content`/`content_hash` -- the same technique `rollback_to_version`
    already uses -- sidesteps both problems while still going through the
    DB's own sync-log (`_log_sync_event`) and FTS (`_update_fts_media`)
    machinery, never a bare hand-written UPDATE that bypasses them. The
    post-ingest hook is still dispatched, just AFTER this transaction
    commits, matching its documented "post-commit" contract.

    This rewrite REPLACES whatever `Media.content` holds, so it first proves
    that content is the meeting's own render (fix C2): with the shipped
    `post_transcribe` default the Library item's content is the ingest's
    offline transcription of `mixed.wav`, which a rename would otherwise
    silently destroy. Re-rendering the transcript with the CURRENT (pre-
    rename) name map and comparing is that proof; an empty render (a missing
    or emptied `transcript.jsonl`, Qodo Q16) is refused for the same reason.
    A refusal touches neither the DB nor `meeting.json`.

    Args:
        db: The `MediaDatabase`.
        media_id: The meeting recording's Library media id.
        cluster_id: The speaker cluster to rename.
        name: The new name; empty removes the entry (back to "Speaker N").

    Returns:
        `SpeakerRenameResult(ok=True)` when both authorities were rewritten,
        or `ok=False` with static copy naming why nothing was touched.

    Raises:
        InputError: `media_id` does not name a live media item.
        ConflictError: The row's optimistic-lock version changed
            concurrently (Media or Transcripts).
    """
    row = db.get_media_by_id(media_id)
    if row is None:
        raise InputError(f"Media item {media_id} not found")
    folder = Path(row["url"]).parent

    meeting = read_meeting_json(folder)
    names = dict(meeting.get("speaker_names") or {})
    segments = _read_meeting_transcript_segments(folder)
    current_content = _render_meeting_transcript(segments, names)
    if not current_content.strip():
        return SpeakerRenameResult(False, RENAME_REFUSED_EMPTY_TRANSCRIPT)
    if (row["content"] or "").strip() != current_content.strip():
        return SpeakerRenameResult(False, RENAME_REFUSED_NOT_MEETING_CONTENT)

    name = normalize_speaker_name(name)
    if name:
        names[cluster_id] = name
    else:
        names.pop(cluster_id, None)
    update_meeting_json(folder, speaker_names=names)

    new_content = _render_meeting_transcript(segments, names)
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()

    media_uuid = row["uuid"]
    current_version = row["version"]
    new_version = current_version + 1
    now = db._get_current_utc_timestamp_str()
    client_id = db.client_id

    with db.transaction() as conn:
        cur = conn.cursor()
        # Make the swap reversible (C2): `rollback_to_version` restores a
        # DocumentVersions row's content into `Media`, but refuses the LATEST
        # version number -- so the state being replaced has to be seeded
        # before the new one is appended, or the very first rename would be
        # unrecoverable. Seed once per item; later renames just append.
        cur.execute("SELECT 1 FROM DocumentVersions WHERE media_id = ? LIMIT 1", (media_id,))
        if cur.fetchone() is None:
            db.create_document_version(media_id=media_id, content=row["content"] or "")
        cur.execute(
            "UPDATE Media SET content=?, content_hash=?, last_modified=?, version=?, client_id=? "
            "WHERE id=? AND version=?",
            (new_content, new_hash, now, new_version, client_id, media_id, current_version),
        )
        if cur.rowcount == 0:
            raise ConflictError(f"Media rename conflict for id={media_id}", media_id)
        db._log_sync_event(
            conn, "Media", media_uuid, "update", new_version,
            {"last_modified": now, "version": new_version, "client_id": client_id, "content": new_content},
        )
        db._update_fts_media(conn, media_id, row["title"], new_content)
        _write_meeting_transcript_row(
            db, conn, media_id, row.get("transcription_model"), new_content, now, client_id,
        )
        db.create_document_version(media_id=media_id, content=new_content)

    dispatch_media_post_ingest(db, media_id, media_uuid)
    return SpeakerRenameResult(True)


def _meeting_speaker_legend_rows(db: Any, media_id: int) -> list[tuple[str, str]]:
    """Return `(cluster_id, display_label)` for every speaker in
    `media_id`'s meeting folder, in first-seen `transcript.jsonl` order --
    the same population the live Meetings screen's Task 7 legend tracks via
    `_note_speaker`, applied here to the whole transcript at once since the
    meeting has already finished."""
    row = db.get_media_by_id(media_id)
    if row is None:
        return []
    folder = Path(row["url"]).parent
    names = dict(read_meeting_json(folder).get("speaker_names") or {})
    segments = _read_meeting_transcript_segments(folder)
    seen: list[str] = []
    for segment in segments:
        # A hand-edited transcript.jsonl can carry a speaker_id that is not a
        # legal Textual widget id ("S 1", "S#1"); interpolating it below would
        # raise out of compose() and take the screen down. Skip the row.
        if not is_widget_safe_cluster_id(segment.speaker_id or ""):
            continue
        if segment.speaker_id not in seen:
            seen.append(segment.speaker_id)
    rows = []
    for cluster_id in seen:
        placeholder = MeetingSegment(0, 0.0, 0.0, 0.0, 0.0, "others", "", speaker_id=cluster_id)
        label = render_label(placeholder, names, _MEETING_USER_DISPLAY_NAME) or cluster_id
        rows.append((cluster_id, label))
    return rows


_MEDIA_ROW_COMPACT_HEIGHT = 1
_MEDIA_ROW_WIDE_HEIGHT = 2

# task-30043 (critique 2026-09-03 P1): the items pane sits at ~40-44 cols in
# EVERY real shell layout (3-pane reading shell AND the compact stage), so a
# single six-button row can never render its labels there -- live capture
# showed ``t so E Tr R Se`` and select mode's bulk actions as bare ``○ ○ ○``.
# The multi-row grammar below is therefore THE grammar, not a responsive
# variant: every row's label sum (including the "○ "-prefixed disabled
# forms) is budgeted to fit the pane's 40-col floor.


@dataclass(frozen=True)
class LibraryMediaRowGeometry:
    """One public Textual geometry revision from a Media row-scroll owner."""

    revision: int
    size: Size
    virtual_size: Size
    container_size: Size | None


class LibraryMediaRowGeometryChanged(Message):
    """Report one concrete Media row-scroll owner's revised geometry."""

    def __init__(
        self,
        owner: "LibraryMediaRowScroll",
        geometry: LibraryMediaRowGeometry,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.geometry = geometry


class LibraryMediaRowScroll(VerticalScroll):
    """Publish distinct Resize-derived geometry for the owning Media list."""

    latest_geometry: LibraryMediaRowGeometry | None = None

    def on_resize(self, event: events.Resize) -> None:
        """Publish distinct, monotonically revised owner geometry after reflow."""
        previous = self.latest_geometry
        geometry_values = (event.size, event.virtual_size, event.container_size)
        if previous is not None and geometry_values == (
            previous.size,
            previous.virtual_size,
            previous.container_size,
        ):
            return
        geometry = LibraryMediaRowGeometry(
            revision=1 if previous is None else previous.revision + 1,
            size=event.size,
            virtual_size=event.virtual_size,
            container_size=event.container_size,
        )
        self.latest_geometry = geometry
        self.post_message(LibraryMediaRowGeometryChanged(self, geometry))


def _capped_choice_value(value: str, cap: int = 8) -> str:
    """Bound a data-derived chooser value for its opener label (task-30043).

    Args:
        value: The stored value (e.g. a media type).
        cap: Maximum characters to show before an ellipsis.

    Returns:
        The value, or its first ``cap - 1`` characters plus ``…``.
    """
    value = str(value)
    return value if len(value) <= cap else value[: cap - 1] + "…"


def _media_row_label_rest(
    title: str,
    secondary: str,
    *,
    compact: bool,
    loading: bool = False,
    loaded: bool = False,
) -> str:
    """Return the marker-free Media row label for one responsive density.

    task-30044 (critique 2026-09-03 P2): both densities use the SHORT state
    prefix ("Loaded · " / "Loading · ") -- the old wide-mode prose ("Loaded
    in Reader            ") consumed ~28 of ~35 label cells and displaced
    titles to "Quart"/"SQLit", so the row that mattered most was the one
    you couldn't identify.
    """
    visible_title = _visible_row_title(title)
    state = "Loading" if loading else "Loaded" if loaded else ""
    prefix = f"{state} · " if state else ""
    if compact:
        return f" {prefix}{visible_title} · {secondary}"
    return f" {prefix}{visible_title}\n    {secondary}"


class LibraryMediaCanvas(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
    """Render the Library media list with a type filter and preview.

    Attributes:
        canvas: Current media canvas display state.
    """

    BUNDLED_CSS = """
    /* task-30043: the multi-row action grammar needs CONTENT-width buttons.
     * Textual Button's 16-cell min-width floor alone would overflow the
     * pane's 40-col floor (six floors = 96 cells); each row's label budget
     * is what keeps task-28025's fit contract true. Baseline geometry lives
     * here so harnesses without the app bundle lay out like the app. */
    .ds-toolbar > .library-canvas-action,
    .ds-toolbar > .library-toolbar-count {
        width: auto;
        min-width: 0;
    }
    /* task-31224: Textual Input defaults to width 100%, which consumed the
     * whole filter row and pushed "Clear filter" off-screen -- the one
     * honest recovery for a filter miss was invisible (live: it never
     * rendered at any width). Share the row instead. */
    #library-media-filter {
        width: 1fr;
    }
    #library-media-filter-clear {
        width: auto;
        min-width: 0;
    }
    /* task-31270: receipts are two rows, full width; the copy wraps and the
     * action row keeps content-width buttons so Undo/Dismiss always paint. */
    .library-media-receipt {
        width: 100%;
        height: auto;
    }
    .library-media-receipt > .library-media-receipt-copy {
        width: 100%;
        height: auto;
    }
    .library-media-receipt > .library-media-receipt-actions {
        width: 100%;
        height: auto;
    }
    /* Task 8: same trap as #library-media-filter above -- Input defaults to
     * width 100%, which inside a row Horizontal would blow the label off
     * to the side. */
    .library-media-speaker-row {
        width: 100%;
        height: auto;
    }
    .library-media-speaker-row Static {
        width: auto;
        min-width: 0;
    }
    .library-media-speaker-row Input {
        width: 1fr;
    }
    """

    def __init__(
        self,
        canvas: LibraryMediaCanvasState,
        *,
        pager: LibraryPagerDisplay | None = None,
        type_options: tuple[str | None, ...] | None = None,
        stale_action_reason: str = "",
        mutation_action_reason: str = "",
        analysis_action_reason: str = "",
        compact: bool = False,
        show_preview: bool = True,
        can_rename_speakers: bool = False,
        media_db: Any = None,
        speaker_rename_media_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas = canvas
        self.pager = pager
        self.type_options = (
            canvas.type_options if type_options is None else type_options
        )
        self.stale_action_reason = stale_action_reason
        self.mutation_action_reason = mutation_action_reason
        self.analysis_action_reason = analysis_action_reason
        self.compact = compact
        self.show_preview = show_preview
        # Task 8 (meeting diarization spec): True only when the selected
        # item's meeting folder still holds a `meeting.json`
        # (`can_rename_meeting_speakers`, computed by the caller). `media_db`
        # + `speaker_rename_media_id` are the real `MediaDatabase` and the
        # selected item's backing id -- needed here (breaking this canvas's
        # otherwise pure-state design on purpose) so the legend below can
        # read the meeting folder and actually call `rename_meeting_speaker`
        # rather than just showing an inert control. Absent, not merely
        # disabled, when False/None -- there is nothing to rename.
        self.can_rename_speakers = can_rename_speakers
        self.media_db = media_db
        self.speaker_rename_media_id = speaker_rename_media_id
        # Fill the (already 13fr) canvas host, not an independent 13fr --
        # ``LibraryMediaViewer`` documented this trap first: an `fr` width
        # here resolves against the HOST's content width per fraction, so
        # 13fr laid this canvas out ~13x wider than visible (measured 1703
        # cols on a 170-col terminal) and children clipped instead of
        # ellipsizing. task-14900's side-by-side split needs the panes to
        # divide the REAL width, so the canvas must be bounded like the
        # viewer already is.
        self.styles.width = "1fr"
        # task-30043: 36, not 40 -- the 3-pane shell actually allots this
        # pane ~37 visible cells, so a 40-cell floor overflowed the slot and
        # silently clipped the rightmost ~3 cells of every child (live: the
        # armed-confirm copy wrapped at 40 and rendered "restore later from
        # Tr"). The floor only exists to bound the 13fr resolution trap
        # below; 36 keeps that while letting the canvas fit its real slot.
        self.styles.min_width = 36

    def sync_state(
        self,
        canvas: LibraryMediaCanvasState,
        *,
        pager: LibraryPagerDisplay | None = None,
        type_options: tuple[str | None, ...] | None = None,
        stale_action_reason: str = "",
        mutation_action_reason: str = "",
        analysis_action_reason: str = "",
        compact: bool = False,
        show_preview: bool = True,
        can_rename_speakers: bool = False,
        media_db: Any = None,
        speaker_rename_media_id: int | None = None,
    ) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest media canvas display state.

        Returns:
            None.
        """
        self.canvas = canvas
        self.pager = pager
        self.type_options = (
            canvas.type_options if type_options is None else type_options
        )
        self.stale_action_reason = stale_action_reason
        self.mutation_action_reason = mutation_action_reason
        self.analysis_action_reason = analysis_action_reason
        self.compact = compact
        self.show_preview = show_preview
        self.can_rename_speakers = can_rename_speakers
        self.media_db = media_db
        self.speaker_rename_media_id = speaker_rename_media_id
        self.refresh(recompose=True)

    # ---- Task 8 (meeting diarization spec): inline speaker rename ---------
    _SPEAKER_INPUT_PREFIX = "library-media-speaker-input-"

    @on(Input.Submitted, "#library-media-speaker-legend Input")
    def _handle_speaker_rename_submitted(self, event: Input.Submitted) -> None:
        """Rename the submitted row's speaker and refresh the shown transcript.

        Mirrors the live Meetings screen's Task 7 `_apply_rename`: the
        rename itself is unconditional (a submit racing teardown should
        still persist), and only the widget refresh afterwards is
        `is_mounted`-guarded.
        """
        event.stop()
        widget_id = event.input.id or ""
        if not widget_id.startswith(self._SPEAKER_INPUT_PREFIX):
            return
        cluster_id = widget_id[len(self._SPEAKER_INPUT_PREFIX):]
        name = normalize_speaker_name(event.value)
        event.input.value = ""
        if self.media_db is None or self.speaker_rename_media_id is None:
            return
        # Qodo Q3: the rename reads the transcript file, runs several DB
        # writes, FTS maintenance and a post-ingest dispatch -- all of which
        # would freeze the UI on a large transcript or a busy database.
        self.run_worker(
            lambda: self._rename_speaker_off_thread(cluster_id, name),
            group="library-media-speaker-rename",
            thread=True,
            exit_on_error=False,
        )

    def _rename_speaker_off_thread(self, cluster_id: str, name: str) -> None:
        """Persist one rename on a worker thread, then refresh on the UI one."""
        try:
            outcome = rename_meeting_speaker(
                self.media_db, self.speaker_rename_media_id, cluster_id, name
            )
        except Exception as exc:  # noqa: BLE001 - a rename must not crash the canvas
            # `rename_meeting_speaker` reads/writes `meeting.json`; a
            # filesystem failure's `str()` embeds the meeting folder path
            # (task-9 diagnostic inventory review) -- redact it, mirroring
            # `meetings_screen.py`'s own rename-persist failure log.
            logger.warning("Library media speaker rename failed: {}", redact_user_paths(str(exc)))
            outcome = SpeakerRenameResult(False, f"unexpected error ({type(exc).__name__})")
        self.app.call_from_thread(self._apply_speaker_rename_outcome, cluster_id, outcome)

    def _apply_speaker_rename_outcome(
        self, cluster_id: str, outcome: SpeakerRenameResult
    ) -> None:
        """Report a refused/failed rename, or repaint after a successful one.

        Qodo Q15: a rename that changed nothing used to leave only a debug
        log, so the user saw the old name and no explanation.
        """
        if not outcome.ok:
            self.app.notify(f"Couldn't rename this speaker: {outcome.reason}.", severity="warning")
            return
        if not self.is_mounted:
            return
        self._refresh_after_speaker_rename(cluster_id)

    def _refresh_after_speaker_rename(self, cluster_id: str) -> None:
        """Re-read the rewritten `Media.content` and patch the preview text
        plus the just-renamed row's own legend label in place."""
        row = self.media_db.get_media_by_id(self.speaker_rename_media_id)
        content = row["content"] if row else ""
        try:
            self.query_one("#library-media-preview-lines", Static).update(content)
        except NoMatches:
            pass
        try:
            speaker_rows = dict(
                _meeting_speaker_legend_rows(self.media_db, self.speaker_rename_media_id)
            )
            label_widget = self.query_one(
                f"#library-media-speaker-label-{cluster_id}", Static
            )
            label_widget.update(speaker_rows.get(cluster_id, cluster_id))
        except Exception:  # noqa: BLE001 - legend label refresh is best-effort
            pass

    def apply_compact_presentation(self, compact: bool) -> None:
        """Patch mounted Media density and preview participation in place."""
        self.compact = compact
        select_mode = getattr(self.canvas, "select_mode", False)
        row_height = (
            _MEDIA_ROW_COMPACT_HEIGHT if compact else _MEDIA_ROW_WIDE_HEIGHT
        )
        for button in self.query(".library-media-row"):
            title = button._library_media_title
            secondary = button._library_media_secondary
            label_rest = _media_row_label_rest(
                title,
                secondary,
                compact=compact,
                loading=button._library_media_loading,
                loaded=button._library_media_loaded,
            )
            button._library_row_label_rest = label_rest
            if select_mode:
                marker = "☑" if button._library_media_checked else "☐"
            else:
                marker = (
                    "▸"
                    if button._library_media_selected and not compact
                    else " "
                )
            button.label = f"{marker}{label_rest}"
            button.set_class(
                button._library_media_selected and not compact and not select_mode,
                "library-media-row-selected",
            )
            button.styles.height = row_height
            button.styles.min_height = row_height
            self._gate_mutation_action(button, label_rest.lstrip())
        try:
            preview = self.query_one("#library-media-preview")
            open_viewer = self.query_one("#library-media-open-viewer", Button)
        except NoMatches:
            return
        preview.display = self.show_preview and self._has_preview and not compact
        open_viewer.can_focus = self.show_preview and not compact

    def apply_reader_state(self, canvas: LibraryMediaCanvasState) -> None:
        """Patch Reader row state without replacing row widgets.

        Args:
            canvas: Fresh media canvas state carrying selection and load flags.
        """
        self.canvas = canvas
        rows = {row.media_id: row for row in canvas.rows}
        select_mode = getattr(canvas, "select_mode", False)
        for button in self.query(".library-media-row"):
            row = rows.get(str(button.media_id))
            if row is None:
                continue
            button._library_media_selected = row.selected
            button._library_media_checked = row.checked
            button._library_media_loading = row.loading
            button._library_media_loaded = row.loaded
            label_rest = _media_row_label_rest(
                row.title,
                row.secondary,
                compact=self.compact,
                loading=row.loading,
                loaded=row.loaded,
            )
            button._library_row_label_rest = label_rest
            if select_mode:
                marker = "☑" if row.checked else "☐"
            else:
                marker = "▸" if row.selected and not self.compact else " "
            button.label = f"{marker}{label_rest}"
            button.set_class(
                row.selected and not self.compact and not select_mode,
                "library-media-row-selected",
            )

    def _gate_stale_action(self, button: Button, base_label: str) -> Button:
        """Apply the controller's stale-OR-mutation gate to one unsafe action.

        Final review M-1: despite the name (and despite reading as the
        symmetric partner of ``_gate_mutation_action`` below, which gates on
        write-in-flight only), this disables on EITHER input -- a stale page
        OR a write actually in flight. Do not assume a mutation ending
        leaves these controls live if the page is still stale.
        """
        reason = self.mutation_action_reason or self.stale_action_reason
        if reason:
            button.label = library_disabled_action_label(base_label, True)
            button.disabled = True
            button.tooltip = reason
        return button

    def _select_all_button(self, rendered_count: int) -> Button:
        """Build the Select-all bulk action (summary-row sibling of the count)."""
        label = f"Select all {rendered_count} shown"
        select_all = Button(
            label,
            id="library-media-select-all",
            classes="library-canvas-action",
            compact=True,
        )
        return self._gate_stale_action(select_all, label)

    def _clear_selection_button(self) -> Button:
        """Build the Clear bulk action."""
        clear = Button(
            "Clear",
            id="library-media-select-clear",
            classes="library-canvas-action",
            compact=True,
        )
        return self._gate_stale_action(clear, "Clear")

    def _bulk_action_button(
        self,
        base: str,
        widget_id: str,
        disabled_tooltip: str,
        enabled_tooltip: str,
        *,
        danger: bool = False,
    ) -> Button:
        """Build one Export/Review/Delete bulk action (task-30043).

        Short REAL words ("Export" / "Review" / "Delete") -- the mode and the
        adjacent count make them unambiguous, the F-018 tooltips carry the
        full sentences, and a disabled action can never collapse to a bare
        "○" marker the way the full labels did at the pane's 40-col floor.

        task-4023 AC#1 (RC-07): "○" disabled marker -- these are the very
        buttons the user entered Select mode looking for, previously
        colour-only at a measured 1.39:1. The base label is stashed so
        `_apply_library_row_toggle`'s in-place patch can rebuild it when
        the selection count crosses 0.
        """
        bulk_disabled = self.canvas.selected_count == 0
        classes = "library-canvas-action"
        if danger:
            classes += " library-media-action-danger"
        button = Button(
            library_disabled_action_label(base, bulk_disabled),
            id=widget_id,
            classes=classes,
            compact=True,
        )
        button._library_disabled_marker_base = base
        button.disabled = bulk_disabled
        # F-018: a disabled action says why.
        button.tooltip = disabled_tooltip if bulk_disabled else enabled_tooltip
        return self._gate_stale_action(button, base)

    def _select_mode_bulk_buttons(self) -> ComposeResult:
        """Yield the Export and Review bulk actions (Delete rides its own row)."""
        yield self._bulk_action_button(
            "Export",
            "library-media-export-selected",
            LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_EXPORT_SELECTED_TOOLTIP,
        )
        # task-28242: "Review selected" pins the selection as an ordered
        # review set.
        yield self._bulk_action_button(
            "Review",
            "library-media-review-selected",
            LIBRARY_REVIEW_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_REVIEW_SELECTED_TOOLTIP,
        )

    def _analyze_selected_button(self) -> Button:
        """Build the "Analyze" bulk action (task-28007 AC#4).

        Rides its OWN row rather than joining Clear/Export/Review: those
        three already measure 33 of the pane's 36 cells (`min_width` above),
        so a fourth 13-cell action clipped every label on that row. Same
        multi-row grammar the danger row uses. When no analysis provider is
        configured the resolver's own sentence replaces the F-018 tooltip,
        so the disabled control says WHY, not just that it is off (AC#5's
        rule, applied to the bulk gesture).
        """
        button = self._bulk_action_button(
            "Analyze",
            "library-media-analyze-selected",
            LIBRARY_ANALYZE_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_ANALYZE_SELECTED_TOOLTIP,
        )
        if self.analysis_action_reason:
            button.label = library_disabled_action_label("Analyze", True)
            button.disabled = True
            button.tooltip = self.analysis_action_reason
        return button

    def _delete_selected_button(self) -> Button:
        """Build the isolated danger action (task-2853's far-end rule, upgraded)."""
        return self._bulk_action_button(
            "Delete",
            "library-media-delete-selected",
            LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_DELETE_SELECTED_TOOLTIP,
            danger=True,
        )

    def _gate_mutation_action(self, button: Button, base_label: str) -> Button:
        """Disable even recovery controls only while a write is unsettled."""
        if self.mutation_action_reason:
            button.label = library_disabled_action_label(base_label, True)
            button.disabled = True
            button.tooltip = self.mutation_action_reason
        return button

    def compose(self) -> ComposeResult:
        """Render the header/filter, status line, media rows, and preview.

        Returns:
            ComposeResult for the media canvas.
        """
        title_count = self.pager.title_count if self.pager is not None else self.canvas.count
        title = "Media" if title_count is None else f"Media ({title_count})"
        select_mode = getattr(self.canvas, "select_mode", False)
        fresh_zero = (
            self.pager is not None
            and title_count == 0
            and not self.canvas.rows
            and not select_mode
            and not self.canvas.delete_receipt_count
            and not self.stale_action_reason
            and not self.mutation_action_reason
            and not self.pager.status_copy
            and not self.pager.retry_visible
        )
        # task-28243: the "Sets" picker opener lives on the TITLE row, not the
        # action toolbar -- that toolbar already overflows the narrow items
        # pane (task-28025) and one more button pushed a squeezed Button into
        # rich's zero-width chop_cells crash (live-verified 2026-09-02). The
        # title row carries ~9 chars in a min-width-40 pane, so both widgets
        # always render at full width. Auto-width Static + fixed compact
        # Button only (task-4023's render-safe grammar: no 1fr sibling).
        # Hidden in select mode like the other list-level actions; on the
        # fresh-empty page it is not composed at all -- that page pins
        # exactly ONE recovery action (and display:none still matches DOM
        # queries).
        title_row = Horizontal(id="library-media-title-row")
        title_row.styles.height = "auto"
        with title_row:
            title_static = Static(title, id="library-media-title")
            # A Static defaults to 1fr inside a Horizontal and would swallow
            # the whole row, pushing the button out of view (live-verified).
            title_static.styles.width = "auto"
            yield title_static
            if not fresh_zero:
                sets_btn = Button(
                    "Sets",
                    id="library-media-review-sets",
                    classes="library-canvas-action",
                    compact=True,
                    tooltip="Resume, switch, or dismiss saved review sets.",
                )
                sets_btn.display = not select_mode
                yield sets_btn
        filter_row = Horizontal(classes="ds-toolbar")
        filter_row.styles.height = "auto"
        with filter_row:
            yield Input(
                value=self.canvas.query,
                # task-31274: say that keywords match too. Kept to 14 cells
                # because Textual word-wraps the placeholder and paints only
                # its first line: at the default 38-col Items pane "Filter by
                # title, content or keyword…" rendered as "Filter by" (live,
                # 235x52). The empty state names the full field set.
                placeholder="Title/keyword…",
                # The placeholder is truncated by design (see above), so the
                # long form lives here rather than nowhere.
                tooltip="Filter by title, content or keyword",
                id="library-media-filter",
            )
            clear_filter = Button(
                "Clear filter",
                id="library-media-filter-clear",
                compact=True,
            )
            clear_filter.disabled = not bool(self.canvas.query)
            yield clear_filter
        if fresh_zero:
            yield Static(
                self.canvas.empty_copy,
                id="library-media-status",
                markup=False,
            )
            # task-31224: a FILTER miss must not suggest importing -- the
            # query-echoing status copy plus the (now visible) Clear filter
            # control above are the honest recovery. Import/Show-all stay
            # the recovery for a genuinely empty source only.
            if self.canvas.query:
                return
            if self.canvas.active_type is None:
                yield Button(
                    "Import media",
                    id="library-media-empty-import",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                yield Button(
                    "Show all types",
                    id="library-media-empty-clear-type",
                    classes="library-canvas-action",
                    compact=True,
                )
            return
        # Gate/label off the RENDERED rows, not ``canvas.count`` -- the latter
        # is the pre-filter total across ALL media types, so with a media-type
        # filter active it overstates what's shown (and stays > 0 when the
        # filter renders nothing). ``handle_library_media_select_all`` already
        # selects only the rendered rows, so this keeps the copy/gate honest.
        # Also portable to the conversations canvas state, which has no
        # ``.count`` field.
        rendered_count = len(self.canvas.rows)
        # task-4023 AC#5: one toolbar grammar across the list canvases --
        # these three actions used to stack VERTICALLY (one full-width
        # button per row) while Notes/Prompts/Skills lay theirs out in
        # horizontal ``ds-toolbar`` rows. Same render-safe shape as those
        # canvases: fixed-width compact Buttons only, never mixed with a
        # 1fr sibling.
        # task-14902: while the type choice strip is open it REPLACES this
        # toolbar row (the Notes Sort precedent -- browse actions hide while
        # the chooser is showing), keeping the vertical budget flat.
        type_choices_visible = getattr(self.canvas, "type_choices_visible", False)
        sort_choices_visible = getattr(self.canvas, "sort_choices_visible", False)
        toolbar_visible = not (type_choices_visible or sort_choices_visible)
        sort_labels = dict(MEDIA_SORT_CHOICES)
        current_sort = getattr(self.canvas, "sort_by", "last_modified_desc")
        type_filter = Button(
            # task-14902: a chooser-opener, no longer a cycler -- press
            # opens the direct-pick strip below instead of advancing.
            # Qodo #2350: the VALUE is data and can be long; the label caps
            # it so the chooser row's budget holds (the tooltip and the
            # chooser strip itself carry the full value).
            library_choice_label(
                "type",
                "All types"
                if self.canvas.active_type is None
                else _capped_choice_value(self.canvas.active_type),
            ),
            id="library-media-type-filter",
            classes="library-canvas-action",
            compact=True,
            tooltip=library_choice_tooltip(
                "media type",
                tuple(
                    "All types" if value is None else value
                    for value in self.type_options
                ),
            ),
        )
        self._gate_mutation_action(type_filter, str(type_filter.label))
        # task-28013: sort chooser opener -- hidden in select mode like
        # Export/Trash (Select's toolbar acts on the selection).
        sort_btn = Button(
            library_choice_label(
                "sort", sort_labels.get(current_sort, "Newest")
            ),
            id="library-media-sort",
            classes="library-canvas-action",
            compact=True,
            tooltip=library_choice_tooltip(
                "the sort order", tuple(label for _, label in MEDIA_SORT_CHOICES)
            ),
        )
        sort_btn.display = not select_mode
        self._gate_stale_action(sort_btn, str(sort_btn.label))
        export_btn = Button(
            "Export…",
            id="library-media-export",
            classes="library-canvas-action",
            compact=True,
        )
        export_btn.display = not select_mode
        self._gate_stale_action(export_btn, "Export…")
        # task-4025: the browsable Trash surface's entry point -- a
        # plain navigation action (never a `type:` cycle value: `type:`
        # cycles CONTENT types derived from the records, and trash is a
        # STATE). Always enabled: the trash count isn't known until its
        # view fetches, and an empty Trash shows its honest empty copy
        # rather than this button lying disabled. Hidden in select mode
        # like "Export…" -- Select's toolbar is for acting on the
        # selection, not navigating away from it.
        trash_btn = Button(
            "Trash",
            id="library-media-trash-open",
            classes="library-canvas-action",
            compact=True,
            tooltip="Browse and restore deleted media.",
        )
        trash_btn.display = not select_mode
        # task-28242: "Review these" pins the WHOLE filtered result as an
        # ordered review set and walks it in the Reader. A list-level
        # action, hidden in select mode like Export/Trash.
        review_btn = Button(
            "Review these",
            id="library-media-review",
            classes="library-canvas-action",
            compact=True,
            tooltip="Review every item in this list, one by one.",
        )
        review_btn.display = not select_mode
        self._gate_stale_action(review_btn, "Review these")
        # Disable only when there's nothing to select AND we're not
        # already in select mode -- in select mode the button is "Done"
        # and must always be pressable so the user can exit even if the
        # rows dropped to zero (e.g. a background snapshot refresh
        # emptied the list).
        select_disabled = rendered_count == 0 and not select_mode
        select_btn = Button(
            # task-4023 AC#1 (RC-07): disabled carries the non-colour
            # "○" marker; the F-018 reason tooltip below says why.
            library_disabled_action_label(
                "Done" if select_mode else "Select", select_disabled
            ),
            id="library-media-select-toggle",
            classes="library-canvas-action",
            compact=True,
        )
        select_btn.disabled = select_disabled
        if select_disabled:
            select_btn.tooltip = LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP
        self._gate_stale_action(
            select_btn, "Done" if select_mode else "Select"
        )
        # task-30043 (critique P1): at the items pane's ~40-col real width
        # one Horizontal cannot render six labels (live capture: ``t so E Tr
        # R Se``), so the browse actions split into rows of readable labels.
        # In select mode most of these hide, so ``type:`` keeps that row to
        # itself and Done moves out of the toolbar entirely (see the Done row
        # at the end of the select-mode block below).
        if not select_mode:
            toolbar_rows: tuple[tuple[str, tuple[Button, ...]], ...] = (
                ("library-media-toolbar-choosers", (type_filter, sort_btn)),
                (
                    "library-media-toolbar-actions",
                    (export_btn, trash_btn, select_btn),
                ),
                ("library-media-toolbar-review", (review_btn,)),
            )
        else:
            toolbar_rows = (
                (
                    "library-media-toolbar",
                    (
                        type_filter,
                        sort_btn,
                        export_btn,
                        trash_btn,
                        review_btn,
                    ),
                ),
            )
        for row_id, row_buttons in toolbar_rows:
            row = Horizontal(id=row_id, classes="ds-toolbar")
            row.styles.height = "auto"
            row.display = toolbar_visible
            with row:
                for button in row_buttons:
                    yield button
        if type_choices_visible:
            options: list[Option] = []
            highlighted = 0
            for index, value in enumerate(self.type_options):
                display = "All types" if value is None else value
                option = Option(
                    f"✓ {display}" if value == self.canvas.active_type else display,
                    id=f"library-media-type-option-{index}",
                )
                option.choice_value = value
                options.append(option)
                if value == self.canvas.active_type:
                    highlighted = index
            choices = OptionList(
                *options,
                id="library-media-type-choices",
                compact=True,
                markup=False,
            )
            choices.highlighted = highlighted
            choices.styles.height = min(8, max(1, len(options)))
            yield choices
        if sort_choices_visible:
            # task-31235 (critique #3 P1): a vertical OptionList exactly like
            # the type chooser above -- the horizontal choice strip clipped
            # "Title A-Z" and rendered "Title Z-A" nowhere at the items
            # pane's real width, while keyboard selection could still pick
            # the invisible option.
            sort_options: list[Option] = []
            sort_highlighted = 0
            for index, (value, label) in enumerate(MEDIA_SORT_CHOICES):
                option = Option(
                    f"✓ {label}" if value == current_sort else label,
                    id=f"library-media-sort-option-{index}",
                )
                option.choice_value = value
                sort_options.append(option)
                if value == current_sort:
                    sort_highlighted = index
            sort_choices = OptionList(
                *sort_options,
                id="library-media-sort-choices",
                compact=True,
                markup=False,
            )
            sort_choices.highlighted = sort_highlighted
            sort_choices.styles.height = min(8, max(1, len(sort_options)))
            yield sort_choices
        confirming_bulk_delete = getattr(self.canvas, "confirming_bulk_delete", False)
        if select_mode:
            if confirming_bulk_delete:
                # A single full-width Static above the toolbar, not inside it
                # -- mixing a long sentence Static with the toolbar's fixed-
                # width Buttons in one Horizontal is the known non-rendering
                # failure mode (see LibraryMediaViewer.compose's delete-
                # confirm copy, the same pattern this mirrors). The short
                # "N selected" Static below is unaffected -- it is already
                # proven to render alongside Buttons in this exact row.
                # task-4025 AC3 (ADR-055 Pattern A): the confirm copy names
                # the durable recovery path -- the Trash view this task
                # built (the list toolbar's own "Trash" action) -- on top
                # of the receipt's immediate Undo. Supersedes task-4022
                # AC3's honest "there's no Trash view" copy, which was
                # true only until this surface existed.
                item_word = "item" if self.canvas.selected_count == 1 else "items"
                confirm_copy = Static(
                    f"Delete {self.canvas.selected_count} selected {item_word}? "
                    "You can undo right away, or restore later from Trash.",
                    id="library-media-bulk-delete-confirm-copy",
                    markup=False,
                )
                # task-30043: bound the copy to the pane so the safety
                # sentence WRAPS -- unbounded, it clipped mid-word ("You can
                # und / restore later from Tr") at the narrow pane width.
                confirm_copy.styles.width = "1fr"
                confirm_copy.styles.height = "auto"
                yield confirm_copy
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                # Bug found via task-2853's OWN live tmux verification
                # (reproduced against pre-task-8 HEAD too, so it predates
                # this task, and against the Conversations canvas too, the
                # identical pattern -- see review round 2): with no
                # explicit width, this Static resolved as unbounded inside
                # the ``ds-toolbar`` ``Horizontal`` -- live capture showed
                # it claiming ~1700 columns on a 170-column terminal,
                # pushing every sibling Button entirely off-screen
                # (invisible, though still present in the DOM -- which is
                # why headless ``query_one`` pilot tests never caught it).
                # Fixed as the general rule via the shared
                # ``library-toolbar-count`` class (css/components/
                # _agentic_terminal.tcss), not a per-widget Python
                # one-off, so every canvas's counter is covered by one
                # declaration.
                yield Static(
                    f"{self.canvas.selected_count} selected",
                    id="library-media-selected-count",
                    classes="library-toolbar-count",
                    markup=False,
                )
                if confirming_bulk_delete:
                    confirm = Button(
                        "Delete",
                        id="library-media-bulk-delete-confirm",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                    yield self._gate_stale_action(confirm, "Delete")
                    cancel = Button(
                        "Cancel",
                        id="library-media-bulk-delete-cancel",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(cancel, "Cancel")
                else:
                    yield self._select_all_button(rendered_count)
            if not confirming_bulk_delete:
                # task-30043: the bulk actions get their own row of short
                # REAL words ("○ Export", never a bare marker) -- one shared
                # row rendered them as ``○ ○ ○`` at the pane's 40-col floor.
                actions_row = Horizontal(
                    id="library-media-select-actions", classes="ds-toolbar"
                )
                actions_row.styles.height = "auto"
                with actions_row:
                    yield self._clear_selection_button()
                    yield from self._select_mode_bulk_buttons()
                # task-28007 AC#4: Analyze gets its own row -- see
                # ``_analyze_selected_button`` for the measurement that put
                # it here rather than beside Export/Review.
                analyze_row = Horizontal(
                    id="library-media-select-analyze", classes="ds-toolbar"
                )
                analyze_row.styles.height = "auto"
                with analyze_row:
                    yield self._analyze_selected_button()
                # task-2853's danger-isolation rule, upgraded: Delete gets a
                # whole row, so it is never adjacent to any other action.
                danger_row = Horizontal(
                    id="library-media-select-danger", classes="ds-toolbar"
                )
                danger_row.styles.height = "auto"
                with danger_row:
                    yield self._delete_selected_button()
            # task-31631 AC#3: Done closes the select-mode block on its own
            # row. It used to ride the toolbar row directly after ``type:``,
            # which is the exact cell range ``sort: Newest`` holds in browse
            # mode (measured at 235x52: Done x=63..71 inside sort's
            # x=63..79), so the habitual click on the sort chooser silently
            # became "leave select mode and discard the selection".
            #
            # Every other slot in the pane's top three rows is likewise a
            # browse control's (type:/sort:, Export…/Trash, Review these),
            # and the "N selected / Select all N shown" summary row already
            # measures 33 of the pane's 36 cells with a two-digit count and
            # a "○ " disabled marker -- a trailing Done clips there. The row
            # AFTER the select-mode block is the only slot that collides
            # with nothing. Rendered outside the ``confirming_bulk_delete``
            # branch above, because Done must stay pressable mid-confirm.
            done_row = Horizontal(
                id="library-media-select-done", classes="ds-toolbar"
            )
            done_row.styles.height = "auto"
            with done_row:
                yield select_btn

        # task-4022 AC2: a completed bulk delete's receipt, naming the
        # count with an Undo affordance right at the point of action --
        # mirrors the ingest queue's own done-row grammar ("✓ done · file
        # · 1s" + a jump action) rather than a toast, which this canvas
        # has none of on the success path today. Rendered OUTSIDE
        # select_mode: a full-success delete exits select mode, so this is
        # the only place left to show it. Uses the same
        # ``library-toolbar-count`` class as "N selected" above -- proven
        # safe for a short Static sharing a ``ds-toolbar`` Horizontal with
        # Buttons (see the comment on that Static; an earlier long-
        # sentence Static in this same row went unbounded and pushed every
        # Button off-screen).
        receipt_count = getattr(self.canvas, "delete_receipt_count", 0)
        if receipt_count:
            receipt_word = "item" if receipt_count == 1 else "items"
            # task-31220 (critique #5): a receipt may only claim success
            # while its Undo can actually run. A failed restore retitles it
            # with the same ✗ glyph the Analyze receipt uses for a run
            # where nothing succeeded, and Undo becomes a retry over the
            # still-failed ids ``receipt_count`` now names.
            undo_failure = getattr(self.canvas, "delete_receipt_undo_failure", "")
            # task-31270 (critique #4 P1): two rows -- copy, then actions --
            # at full width. A single content-width Horizontal clipped Undo
            # to "Und" at the Items pane's real width; same multi-row
            # grammar as the toolbars (task-30043).
            receipt = Vertical(
                id="library-media-bulk-delete-receipt",
                classes="library-media-receipt",
            )
            receipt.styles.height = "auto"
            with receipt:
                yield Static(
                    # task-4025 (ADR-055 Pattern A): the receipt names the
                    # durable path too -- "· in Trash" points at the Trash
                    # view that outlives this receipt's Undo/Dismiss.
                    f"✗ undo failed · {undo_failure}"
                    if undo_failure
                    else f"✓ deleted · {receipt_count} {receipt_word} · in Trash",
                    id="library-media-bulk-delete-receipt-copy",
                    classes="library-toolbar-count library-media-receipt-copy",
                    markup=False,
                )
                actions = Horizontal(
                    classes="ds-toolbar library-media-receipt-actions"
                )
                actions.styles.height = "auto"
                with actions:
                    # task-31220: NOT ``_gate_stale_action``. Undo restores
                    # exactly the ids this receipt names, so it is the
                    # receipt's own recovery and a stale PAGE behind it
                    # cannot invalidate it -- disabling it here broke the
                    # confirmation's "You can undo right away" promise at
                    # the one moment it mattered (critique #5). The shared
                    # write interlock still applies, so a second mutation
                    # can never be claimed while one is in flight.
                    undo_label = "Retry undo" if undo_failure else "Undo"
                    undo = Button(
                        undo_label,
                        id="library-media-bulk-delete-undo",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(undo, undo_label)
                    dismiss = Button(
                        "Dismiss",
                        id="library-media-bulk-delete-receipt-dismiss",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(dismiss, "Dismiss")

        # task-31236: a dismissed review set's undo receipt -- the same
        # grammar as the bulk-delete receipt above, because a one-click
        # dismissal of a mid-walk set (with its done-marks) must be
        # recoverable right where the user lands after the picker closes.
        dismissed_set_name = getattr(
            self.canvas, "review_dismiss_receipt_name", ""
        )
        if dismissed_set_name:
            dismiss_receipt = Vertical(
                id="library-media-review-dismiss-receipt",
                classes="library-media-receipt",
            )
            dismiss_receipt.styles.height = "auto"
            with dismiss_receipt:
                yield Static(
                    f"✓ dismissed · {dismissed_set_name}",
                    id="library-media-review-dismiss-receipt-copy",
                    classes="library-toolbar-count library-media-receipt-copy",
                    markup=False,
                )
                set_actions = Horizontal(
                    classes="ds-toolbar library-media-receipt-actions"
                )
                set_actions.styles.height = "auto"
                with set_actions:
                    # Final review I-3: NOT ``_gate_stale_action``, for the
                    # same reason the bulk-delete receipt's Undo above is
                    # exempt -- this Undo restores exactly the one set its
                    # own copy names, so a stale PAGE behind it cannot
                    # invalidate it. Before this branch both receipts' Undo
                    # were gated identically; leaving this one on the stale
                    # gate let it sit disabled beside a live sibling
                    # receipt's Undo with no rule the user could infer.
                    undo_set = Button(
                        "Undo",
                        id="library-media-review-dismiss-undo",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(undo_set, "Undo")
                    close_receipt = Button(
                        "Dismiss",
                        id="library-media-review-dismiss-receipt-close",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(close_receipt, "Dismiss")

        # task-28007 AC#3/AC#4: the bulk-Analyze receipt -- PR A's two-row
        # grammar (copy, then actions) again, because a set-level run must
        # report per-item outcomes where the gesture happened rather than
        # in a toast that outlives nothing. Three states, one block: the
        # AC#3 Skip/Overwrite choice (nothing has run yet), the live run,
        # and the settled run with Retry failed/Dismiss.
        analyze_choice = getattr(self.canvas, "analyze_choice_count", 0)
        analyze_total = getattr(self.canvas, "analyze_receipt_total", 0)
        if analyze_choice or analyze_total:
            analyze_done = getattr(self.canvas, "analyze_receipt_done", 0)
            analyze_failed = getattr(self.canvas, "analyze_receipt_failed", 0)
            analyze_running = getattr(self.canvas, "analyze_receipt_running", False)
            failed_copy = f" · {analyze_failed} failed" if analyze_failed else ""
            if analyze_choice:
                # R3: no dangling dash -- the buttons are on the row BELOW,
                # so a trailing "— " pointed at nothing. ``analyze_total``
                # is the pressed selection's own size on this path.
                analyze_copy = (
                    f"{analyze_choice} of {analyze_total} already analyzed"
                )
            elif analyze_running:
                # 1-based position of the item being analyzed right now.
                position = min(analyze_done + analyze_failed + 1, analyze_total)
                analyze_copy = f"Analyzing {position} of {analyze_total}{failed_copy}"
            else:
                # A run where NOTHING succeeded must not lead with a tick.
                # ✗ (U+2717) is this repo's failure glyph (see
                # UI/Evals/library_rail.py), paired with the ✓ it
                # replaces here.
                glyph = "✓" if analyze_done else "✗"
                analyze_copy = (
                    f"{glyph} analyzed · {analyze_done} of {analyze_total}"
                    f"{failed_copy}"
                )
            analyze_receipt = Vertical(
                id="library-media-analyze-receipt",
                classes="library-media-receipt",
            )
            analyze_receipt.styles.height = "auto"
            with analyze_receipt:
                yield Static(
                    analyze_copy,
                    id="library-media-analyze-receipt-copy",
                    classes="library-toolbar-count library-media-receipt-copy",
                    markup=False,
                )
                analyze_actions = Horizontal(
                    classes="ds-toolbar library-media-receipt-actions"
                )
                analyze_actions.styles.height = "auto"
                with analyze_actions:
                    if analyze_choice:
                        skip = Button(
                            "Skip them",
                            id="library-media-analyze-skip",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_stale_action(skip, "Skip them")
                        overwrite = Button(
                            "Overwrite",
                            id="library-media-analyze-overwrite",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_stale_action(overwrite, "Overwrite")
                    elif not analyze_running and analyze_failed:
                        retry = Button(
                            "Retry failed",
                            id="library-media-analyze-retry",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_stale_action(retry, "Retry failed")
                    if not analyze_running and not analyze_choice:
                        # A run in flight has nothing to dismiss yet: the
                        # counts are still moving and Dismiss would either
                        # lie (the run continues) or imply a cancel this
                        # gesture does not offer.
                        #
                        # (final review, I-1) The armed CHOICE has no
                        # Dismiss either: three 13-cell buttons overflow
                        # the Items pane at its 36-cell floor (the row
                        # painted "Skip them  Overwrite  Dism", the same
                        # clipping task-31270 fixed for "Und"), and
                        # "Skip them" already IS the change-nothing
                        # outcome -- it retires the card when there is
                        # nothing left to run.
                        analyze_dismiss = Button(
                            "Dismiss",
                            id="library-media-analyze-receipt-dismiss",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_mutation_action(analyze_dismiss, "Dismiss")

        status_text = (
            self.pager.status_copy
            if self.pager is not None and self.pager.status_copy
            else self.canvas.status_copy or self.canvas.empty_copy
        )
        status = Static(
            status_text,
            id="library-media-status",
            markup=False,
        )
        status.display = bool(status_text)
        yield status

        # task-2853 AC4: while Select mode is active, the preview must never
        # show an item outside the current (multi-item) selection context --
        # ``canvas.selected_id``/``preview_lines`` still carry whatever was
        # focused before Select was entered (the UAT's "bottom preview pane
        # meanwhile shows a previously-selected different item" finding), so
        # the whole block is hidden entirely rather than tracking a second,
        # separate "focused row" concept select mode has no use for.
        has_preview = self.show_preview and (
            not select_mode
            and bool(self.canvas.selected_id and self.canvas.preview_lines)
        )
        self._has_preview = has_preview

        # task-14900: the list and its preview share a workbench container
        # (Collections' `#library-collections-workbench` grammar). Above the
        # screen's one measured width regime it lays them out side by side
        # (this Horizontal's default); below it, the host's existing
        # `library-notes-compact` class gives the list the full canvas and
        # suppresses the preview via CSS -- the conditional is keyed off a
        # class the screen already maintains at compose time AND on every
        # resize crossing, so no compose branch here can drift from an
        # in-place updater. Geometry (heights/overflow) moved from inline
        # styles into the same CSS tiers, because inline styles outrank the
        # class-flipped rules.
        workbench = Horizontal(id="library-media-workbench")
        workbench.set_class(has_preview, "has-preview")
        with workbench:
            media_list = Vertical(id="library-media-list")
            with media_list:
                with LibraryMediaRowScroll(id="library-media-row-scroll"):
                    row_height = (
                        _MEDIA_ROW_COMPACT_HEIGHT
                        if self.compact
                        else _MEDIA_ROW_WIDE_HEIGHT
                    )
                    for index, row in enumerate(self.canvas.rows):
                        if select_mode:
                            marker = "☑" if row.checked else "☐"
                        else:
                            marker = "▸" if row.selected and not self.compact else " "
                        # task-281 (PR #665 review): the in-place toggle needs the
                        # marker-less RAW label to rebuild from -- reading it back
                        # off the mounted Button un-escapes user titles (both
                        # ``.plain`` and Textual 8's ``str(Content)`` return
                        # rendered text), so the raw remainder is stashed here at
                        # the single point of truth.
                        label_rest = _media_row_label_rest(
                            row.title,
                            row.secondary,
                            compact=self.compact,
                            loading=row.loading,
                            loaded=row.loaded,
                        )
                        button = Button(
                            f"{marker}{label_rest}",
                            id=f"library-media-row-{index}",
                            classes="library-media-row",
                            compact=True,
                        )
                        button.media_id = row.media_id
                        button._library_row_label_rest = label_rest
                        button._library_media_title = row.title
                        button._library_media_secondary = row.secondary
                        button._library_media_selected = row.selected
                        button._library_media_checked = row.checked
                        button._library_media_loading = row.loading
                        button._library_media_loaded = row.loaded
                        button.tooltip = escape_markup(row.title)
                        # task-31631 AC#2: the whole row is the toggle
                        # target. It already was one full-width Button
                        # ("☐ <title>"), but Textual's ``Button._on_click``
                        # DROPS any click landing while the previous press's
                        # 0.2s ``-active`` flash is still on the widget --
                        # so clicking ☐ and then the same row's title (what
                        # critique #5 did) lost the second click, and the row
                        # read as a one-cell target. A list row has no use
                        # for a press flash; the marker flip is the feedback.
                        # This applies in browse mode too (not just select
                        # mode): dropping the flash there also stops a fast
                        # double-click on a browse row from being swallowed,
                        # and browse-mode feedback is the item loading into
                        # the Reader, not the flash.
                        button.active_effect_duration = 0
                        button.set_class(
                            row.selected and not self.compact and not select_mode,
                            "library-media-row-selected",
                        )
                        button.styles.height = row_height
                        button.styles.min_height = row_height
                        # task-31220: a row OPEN is a read, so it is gated
                        # only while a write is actually unsettled -- never by
                        # the stale gate the open is how you recover from.
                        # Only the mutating actions (Select/Export/sort/
                        # Delete/Undo) stay behind ``_gate_stale_action``.
                        yield self._gate_mutation_action(
                            button, label_rest.lstrip()
                        )
                if self.pager is not None:
                    yield from self._compose_pager(self.pager)

            preview = Vertical(id="library-media-preview")
            preview.display = has_preview and not self.compact
            with preview:
                yield Static(
                    "\n".join(self.canvas.preview_lines),
                    id="library-media-preview-lines",
                    markup=False,
                )
                toolbar = Horizontal(classes="ds-toolbar")
                toolbar.styles.height = "auto"
                with toolbar:
                    # Opens the selected item in the IN-LIBRARY media viewer
                    # (nav stays on Library), distinct from the full viewer's
                    # own action row (`#library-media-open`, `LibraryMediaViewer`
                    # -- "Open in Library ▸ Media", task-2857), which posts a
                    # fresh ``NavigateToScreen`` for the "media" route.
                    open_viewer = Button(
                        "Open in viewer",
                        id="library-media-open-viewer",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    open_viewer.can_focus = self.show_preview and not self.compact
                    yield self._gate_stale_action(open_viewer, "Open in viewer")

                # Task 8 (meeting diarization spec): a finished meeting
                # recording's speakers can still be renamed after the fact --
                # one legend row per speaker (mirroring the live Meetings
                # screen's Task 7 legend), surfaced ONLY while the caller's
                # `can_rename_meeting_speakers` says the meeting folder is
                # still there (absent, not disabled, otherwise: a non-meeting
                # item has nothing to rename).
                # ... and only while the pane is actually visible: building
                # the rows parses the whole transcript.jsonl, which is pure
                # waste on a hidden pane (final review, MINOR).
                if (
                    self.can_rename_speakers
                    and has_preview
                    and not self.compact
                    and self.media_db is not None
                    and self.speaker_rename_media_id is not None
                ):
                    try:
                        speaker_rows = _meeting_speaker_legend_rows(
                            self.media_db, self.speaker_rename_media_id
                        )
                    except Exception:  # noqa: BLE001 - a bad read just means no legend
                        speaker_rows = []
                    if speaker_rows:
                        legend = Vertical(id="library-media-speaker-legend")
                        with legend:
                            for cluster_id, label in speaker_rows:
                                yield Horizontal(
                                    Static(
                                        label,
                                        id=f"library-media-speaker-label-{cluster_id}",
                                        markup=False,
                                    ),
                                    Input(
                                        placeholder="Rename…",
                                        id=f"library-media-speaker-input-{cluster_id}",
                                    ),
                                    classes="library-media-speaker-row",
                                )

            # task-14900: the wide split's detail half never sits blank --
            # when the preview is hidden (Select mode, or an empty list) a
            # placeholder explains the pane, Collections' own detail-pane
            # grammar ("No Collection selected."). CSS-only visibility
            # (never a Python ``display`` write, which would outrank the
            # compact rule that hides it in the preserved stacked layout):
            # hidden while the workbench carries ``has-preview``, and hidden
            # entirely below the breakpoint.
            detail_empty = Static(
                (
                    "No preview in Select mode."
                    if select_mode
                    else "No media item selected."
                ),
                id="library-media-detail-empty",
                markup=False,
            )
            detail_empty.display = self.show_preview
            yield detail_empty

    def _compose_pager(self, pager: LibraryPagerDisplay) -> ComposeResult:
        """Render the controller-owned Media pager below the row viewport."""
        # task-28016: a single-page result has nowhere to page to, so the
        # "Page 1 of 1" counter and the boundary reasons ("Already on the
        # first page.", "No more results.") are pure noise. Show only the item
        # range and keep the (disabled) controls; both return the moment a
        # second page exists.
        disabled_reasons = (
            ()
            if pager.single_page
            else tuple(
                dict.fromkeys(
                    reason
                    for disabled, reason in (
                        (pager.previous_disabled, pager.previous_reason),
                        (pager.next_disabled, pager.next_reason),
                    )
                    if disabled and reason
                )
            )
        )
        status_parts = (
            (pager.range_copy,)
            if pager.single_page
            else (pager.range_copy, pager.page_copy)
        )
        with Vertical(id="library-media-pager", classes="library-source-pager"):
            yield Static(
                " · ".join(copy for copy in status_parts if copy),
                id="library-media-page-status",
                classes="library-source-pager-status",
                markup=False,
            )
            if disabled_reasons:
                yield Static(
                    " · ".join(disabled_reasons),
                    id="library-media-disabled-reason",
                    classes="library-source-pager-status",
                    markup=False,
                )
            # task-31237 (supersedes task-28016's keep-the-disabled-controls
            # choice, critique #3 ruling): a single-page result renders NO
            # pager controls -- two dead "○ Previous ○ Next" forms under
            # every short list were pure noise. The range Static above
            # stays; the controls return the moment a second page exists.
            # A failed fetch still needs its Retry even on one page.
            if pager.single_page and not pager.retry_visible:
                return
            with Horizontal(classes="library-source-pager-controls"):
                previous = Button(
                    library_disabled_action_label(
                        "Previous", pager.previous_disabled
                    ),
                    id="library-media-previous",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.previous_disabled,
                )
                if pager.previous_disabled:
                    previous.tooltip = pager.previous_reason
                yield self._gate_mutation_action(previous, "Previous")
                if pager.retry_visible:
                    retry = Button(
                        "Retry",
                        id="library-media-retry",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(retry, "Retry")
                next_page = Button(
                    library_disabled_action_label("Next", pager.next_disabled),
                    id="library-media-next",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.next_disabled,
                )
                if pager.next_disabled:
                    next_page.tooltip = pager.next_reason
                yield self._gate_mutation_action(next_page, "Next")
