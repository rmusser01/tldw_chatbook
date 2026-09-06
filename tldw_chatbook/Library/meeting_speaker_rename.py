"""After-the-fact meeting speaker rename for a finished meeting's Library item.

Moved out of ``Widgets/Library/library_media_canvas.py`` UNCHANGED
(TASK-31745): the same data layer now backs two surfaces -- the canvas's
(hidden) preview legend and the live media reader's -- so it belongs beside
the other Library logic rather than inside one widget module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from tldw_chatbook.DB.Client_Media_DB_v2 import (
    ConflictError,
    InputError,
    dispatch_media_post_ingest,
)


# ---- Task 8 (meeting diarization spec): rename speakers after the meeting --
# A finished meeting is an ordinary Library Media item; its recording folder
# (holding `meeting.json`'s name map and `transcript.jsonl`'s per-segment
# records) is reachable for free as `Media.url`'s parent directory (`url` is
# the `mixed.wav` path). Both survive raw-track cleanup.
#
# task 31746: the mic channel's display name is read from THIS meeting's own
# `meeting.json` (`user_display_name`, back-filled to "You" by
# `read_meeting_json` for old recordings) rather than a hardcoded literal, so
# this render agrees with what the live Meetings screen showed.


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


def _render_meeting_transcript(
    segments: list[MeetingSegment], names: dict[str, str], user_display_name: str, diarize_mic: bool = False,
) -> str:
    """Render `segments` as the same "[hh:mm:ss] Label: text" lines the live
    meeting screen shows, using the CURRENT name map (task 2's `render_label`)
    and the meeting's own mic-channel display name (task 31746), honouring
    the meeting's own `diarize_mic_channel` flag (task 31743)."""
    lines = []
    for segment in segments:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        label = render_label(segment, names, user_display_name, diarize_mic=diarize_mic)
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

    # `read_meeting_json` only back-fills `speaker_names`/`user_display_name`
    # when `meeting.json` EXISTS -- a missing file (already ruled out for the
    # real UI path by `can_rename_meeting_speakers`, but not enforced here)
    # returns `{}` with neither key, so these `.get(..., default)` calls are
    # not redundant with that back-fill (task 31746 review, item 3).
    meeting = read_meeting_json(folder)
    names = dict(meeting.get("speaker_names") or {})
    user_display_name = meeting.get("user_display_name", "You")
    diarize_mic = meeting.get("diarize_mic_channel", False)
    segments = _read_meeting_transcript_segments(folder)
    current_content = _render_meeting_transcript(segments, names, user_display_name, diarize_mic)
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

    new_content = _render_meeting_transcript(segments, names, user_display_name, diarize_mic)
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
    meeting = read_meeting_json(folder)
    names = dict(meeting.get("speaker_names") or {})
    user_display_name = meeting.get("user_display_name", "You")
    diarize_mic = meeting.get("diarize_mic_channel", False)
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
        label = render_label(placeholder, names, user_display_name, diarize_mic=diarize_mic) or cluster_id
        rows.append((cluster_id, label))
    return rows
