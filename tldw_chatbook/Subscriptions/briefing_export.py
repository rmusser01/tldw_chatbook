"""Export a briefing as markdown, and a watchlist's audio as a podcast feed
directory (spec #2 phase 3, Tasks 1 and 4).

**Markdown half (Task 1).** The UI's `#artifacts-toolbar` Export button
(`UI/Watchlists_Modules/artifacts_pane.py`) lets a user save the SELECTED
briefing's body to a file of their own choosing, through a `FileSave`
dialog (`WatchlistsCollectionsScreen.handle_export_briefing_requested`).
`briefing_markdown_document`/`default_briefing_filename` are the pure half
of that flow: turning a `briefings` row into the document text, and
turning arbitrary (user- or model-authored) text into a filesystem-safe
name. They do no I/O of their own -- the screen validates the chosen
destination (`Utils.path_validation.validate_path_simple`) and writes the
file, off the event loop, itself.

`safe_export_stem` is reused verbatim by the feed half below for episode
filenames, so its contract (alnum/space/-/_ only, a caller-supplied
fallback when nothing survives) is load-bearing for both callers, not just
this one.

**Feed half (Task 4).** `export_feed_directory` copies a watchlist's audio
episodes out of the private `briefing_audio_dir()` into a directory the
user chose (through Task 5's `ExportFeedRequested` flow), alongside a
`feed.xml` RSS document built by `briefing_feed.build_feed_xml`. Unlike the
markdown half above, this DOES do its own I/O -- validating the
destination, copying files, and writing `feed.xml` atomically -- because
the deliverable here is several files landing together, not one screen
writing one file. Three decisions from the phase 3 plan are load-bearing
for this half specifically:

1. **Never route the destination through `Utils/private_paths.py`.**
   `secure_private_directory(..., application_owned=True)` chmods its
   target to `0o700` and raises when not application-owned;
   `create_private_text` opens `"xb"` and refuses to overwrite. Both are
   for APPLICATION-owned storage (`briefing_audio_dir()` itself is exactly
   such a caller). The export destination is the user's OWN folder --
   applying either helper to it would chmod a directory that isn't ours to
   chmod, and fail every re-export. `Utils.path_validation.
   validate_path_simple` plus plain stdlib writes are used instead, the
   same split `UI/STTS_Window.py:4040` and `UI/library_screen.py:6486`
   already make for other user-chosen destinations.
2. **`audio_file_path_is_safe` (imported from `Subscriptions.briefing_audio`,
   not re-derived) runs BEFORE any filesystem access on a `file_path` read
   from the DB.** Phase 2b's Qodo round established this order for playback
   (`artifacts_pane.py`'s own `handle_play_audio_requested`) precisely
   because nothing enforces at the schema level that a stored path is
   safe. Export copies a file OUT of private storage, so getting this
   order wrong here is a wider hole than the playback bug was -- see
   `export_feed_directory`'s own per-episode loop, where the safety check
   is the very first thing done with `file_path`, before `Path(...)`,
   `.exists()`, or any copy. `audio_file_path_is_safe` itself lives beside
   `briefing_audio_dir()` in `briefing_audio.py` (moved there from
   `artifacts_pane.py` in this task's review round 1, since a UI-module
   home made this file's import of it the only `Subscriptions -> UI`
   import in the package); `artifacts_pane` imports it from there too, so
   there is exactly one definition for every caller.
3. **One bad episode is skipped, not fatal.** An unsafe path or a vanished
   source file is recorded as a reason string in `FeedExportResult.skipped`
   and the rest of the export continues -- CLAUDE.md's Error-handling ethos
   (silence is never a state) applied to "one of many episodes broke",
   which is an honest partial success, not a reason to hand the user
   nothing at all.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from ..Utils.path_validation import validate_filename, validate_path_simple
from .briefing_audio import audio_file_path_is_safe
from .briefing_feed import FeedEpisode, build_feed_xml


class BriefingExportError(RuntimeError):
    """Raised when a briefing has nothing worth exporting.

    The only raise site today is a `complete` row whose `body_markdown` is
    NULL or blank -- `briefing_service.generate_briefing` never records an
    empty body as `complete` (an empty provider response is recorded
    `failed` instead), so reaching this is a hand-edited or otherwise
    corrupted row, not a code path the service can produce. An empty file
    is not an export, so this refuses rather than writing nothing and
    calling it done.
    """


def safe_export_stem(text: str, *, fallback: str) -> str:
    """A filesystem-safe filename stem from arbitrary text.

    Whitelists alnum, space, `-` and `_` and drops everything else --
    including path separators, `..`, and any markup-shaped punctuation
    (`[`, `]`, `<`, `>`, `:`, quotes) a watchlist name or briefing title
    might contain. This is a whitelist, not a blacklist of "dangerous"
    characters, for the same reason `content_pane.render_article` gives for
    never hand-escaping model/source text: a blacklist fails open on
    whatever it did not anticipate, while a whitelist cannot.

    `text` may be empty, whitespace-only, or entirely made of characters
    outside the whitelist (e.g. `"###???"`, or a lone `".."`) -- any of
    which leaves nothing to build a name from, so `fallback` is returned
    verbatim in that case. `fallback` is trusted, caller-supplied plain
    text (every caller passes a short literal), so it is not itself run
    through the whitelist.

    Args:
        text: The candidate text -- a watchlist name, a briefing title, or
            similar user/model-authored free text.
        fallback: Returned verbatim when nothing in `text` survives the
            whitelist.

    Returns:
        A non-empty stem containing only alnum, space, `-` and `_`, with
        leading/trailing whitespace stripped.
    """
    cleaned = "".join(
        char for char in (text or "") if char.isalnum() or char in (" ", "-", "_")
    ).strip()
    return cleaned or fallback


def _coverage_window(briefing: Mapping[str, Any]) -> str:
    """What the briefing says it covers, in one line.

    Mirrors `artifacts_pane._window_text`'s own two-part shape (the
    `covers_from_ts` floor and the `covers_through_item_id` watermark), but
    is not that function: this module imports nothing from `UI/` --
    `audio_file_path_is_safe` (used by the feed half below) lives in
    `Subscriptions.briefing_audio`, not in `artifacts_pane`, precisely so
    that stays true (see the module docstring's "Feed half" decision 2).
    `_window_text` itself is a private UI helper, not a shared security
    check, so the string is rebuilt here from the same two columns instead
    of importing it.
    """
    parts: list[str] = []
    covers_from = briefing.get("covers_from_ts")
    if covers_from:
        parts.append(f"since {covers_from}")
    covers_through = briefing.get("covers_through_item_id")
    if covers_through not in (None, ""):
        parts.append(f"through item {covers_through}")
    return " · ".join(parts) if parts else "unknown"


def briefing_markdown_document(briefing: Mapping[str, Any]) -> str:
    """One briefing's body, as a standalone markdown document.

    A short front-matter header precedes the body verbatim, naming the
    four things a reader needs to place the document without the app
    around it: which watchlist it is from (`briefing["watchlist_name"]` --
    a `briefings` row itself only carries `watchlist_id`, so the caller is
    responsible for resolving and merging in the display name before
    calling this, exactly as `WatchlistsCollectionsScreen._watchlist_
    display_name` already does for every other briefing-scoped toast),
    its status, the window it covers, and when it was created.

    Args:
        briefing: A `briefings` row (as `dict(sqlite3.Row)`, or an
            equivalent mapping in tests), with `watchlist_name` merged in.

    Returns:
        The full document text, ready to write to a `.md` file verbatim.

    Raises:
        BriefingExportError: `briefing["body_markdown"]` is NULL, missing,
            or blank. Named error message includes the briefing's id so a
            toast built from `str(exc)` tells the user which row failed.
    """
    body = str(briefing.get("body_markdown") or "").strip()
    if not body:
        raise BriefingExportError(
            f"Briefing {briefing.get('id', 'unknown')} has no body to export."
        )

    watchlist_name = str(briefing.get("watchlist_name") or "").strip() or (
        "this watchlist"
    )
    status = str(briefing.get("status") or "").strip() or "unknown"
    created_at = str(briefing.get("created_at") or "").strip() or "unknown time"
    coverage = _coverage_window(briefing)

    front_matter = (
        "---\n"
        f"watchlist: {watchlist_name}\n"
        f"status: {status}\n"
        f"covers: {coverage}\n"
        f"created: {created_at}\n"
        "---\n\n"
    )
    return f"{front_matter}{body}\n"


def default_briefing_filename(
    briefing: Mapping[str, Any], *, watchlist_name: str
) -> str:
    """The filename a `FileSave` dialog opens with for this briefing.

    Built from the watchlist's name and the briefing's own timestamp, run
    through `safe_export_stem` so a watchlist named with path-shaped or
    markup-shaped text (see that function's own docstring) cannot escape
    the destination directory the user picks in the dialog, nor produce a
    stem with no visible characters at all.

    Args:
        briefing: A `briefings` row -- only `id`/`created_at` are read.
        watchlist_name: The watchlist's display name (resolved by the
            caller; a `briefings` row itself only carries `watchlist_id`).

    Returns:
        `"<stem>.md"`. The stem never contains `/` or `\\` (excluded by
        `safe_export_stem`'s whitelist), so this is always a bare filename,
        never a path.
    """
    created_at = str(briefing.get("created_at") or "").strip()
    stem_source = f"{watchlist_name} {created_at}".strip() if created_at else (
        watchlist_name
    )
    briefing_id = briefing.get("id")
    fallback = f"briefing-{briefing_id}" if briefing_id is not None else "briefing"
    stem = safe_export_stem(stem_source, fallback=fallback)
    return f"{stem}.md"


# --- Feed half (Task 4): writing the feed directory ------------------------

#: `feed.xml`'s own atomic-write pair -- see `_write_feed_xml_atomically`.
_FEED_XML_NAME = "feed.xml"
_FEED_XML_PARTIAL_NAME = "feed.xml.partial"

#: Task 4 review round 1: the mode every file this module WRITES into the
#: user's export directory ends up at -- an ordinary, group/other-readable
#: file, deliberately NOT the `0o600` private-storage mode `briefing_audio_
#: dir()`'s own contents carry. Applied explicitly (not left to the
#: process umask) so the result is deterministic regardless of the
#: caller's environment: `feed.xml`'s partial is created with this mode
#: directly, and each copied episode file is `chmod`ed to it after
#: `shutil.copy2` (which would otherwise carry the PRIVATE source file's
#: mode straight into the user's folder -- see `_copy_episode_audio_file`).
#: A directory the user means to sync, zip, or serve must be readable by
#: more than just their own account; Decision 1 already makes this promise
#: for the directory itself, this pins the same promise for its files.
_EXPORTED_FILE_MODE = 0o644


@dataclass(frozen=True)
class FeedExportResult:
    """The outcome of one `export_feed_directory` call.

    A partial export (some episodes skipped) is a successful result, not an
    exception -- see the module docstring's numbered decision 3. Only a
    failure that leaves the directory with no usable `feed.xml` at all (a
    bad destination, or the atomic write itself failing) raises.

    Attributes:
        directory: The validated, resolved destination directory the feed
            was written into.
        episode_count: How many episodes made it into `feed.xml`.
        skipped: One human-readable reason per episode that did NOT make
            it in -- naming the episode's `audio_id` and why (an unsafe
            path, a vanished file, or a copy failure).
    """

    directory: Path
    episode_count: int
    skipped: list[str]


def _episode_filename(row: Mapping[str, Any], source_path: Path) -> str:
    """The exported filename for one episode's audio file.

    `safe_export_stem` (above, Task 1) turns the row's `preset_name` into a
    filesystem-safe stem; the `audio_id` is always appended so two
    briefings that happen to share a preset name -- or both fall back to
    `safe_export_stem`'s fallback -- can never collide on disk.

    Args:
        row: One `list_watchlist_audio_episodes` row.
        source_path: The episode's audio file, already confirmed safe and
            present. Only its suffix (extension) is reused.

    Returns:
        A bare filename (no path separator), e.g. `"two-host-debate-42.wav"`.
    """
    audio_id = row.get("audio_id")
    stem = safe_export_stem(
        str(row.get("preset_name") or ""), fallback=f"episode-{audio_id}"
    )
    suffix = source_path.suffix or ".wav"
    return f"{stem}-{audio_id}{suffix}"


def _episode_title(row: Mapping[str, Any], published: datetime) -> str:
    """One episode's `<title>`.

    Task 4 review round 1: `preset_name` alone (the original choice) makes
    every episode rendered from the same preset identical in a podcast
    client's episode list, distinguishable only by whatever date field that
    client happens to expose in its list view -- not a safe bet across
    clients. Leading with the publish date here fixes that directly: two
    episodes on the same preset now read as e.g. `"Jan 01, 2026 · Two Host
    Debate"` and `"Jan 02, 2026 · Two Host Debate"`.

    Args:
        row: One `list_watchlist_audio_episodes` row.
        published: The episode's parsed publish timestamp -- the same value
            passed to `FeedEpisode.published`, so the date shown in the
            title always matches the date the feed itself claims.

    Returns:
        A one-line title, date first.
    """
    date_text = published.strftime("%b %d, %Y")
    preset_name = str(row.get("preset_name") or "").strip()
    if preset_name:
        return f"{date_text} · {preset_name}"
    return f"{date_text} · Episode {row.get('audio_id')}"


def _episode_description(row: Mapping[str, Any]) -> str:
    """One episode's `<description>` text.

    Mirrors `_coverage_window`'s "a status IS the observability" spirit at
    episode grain: a listener choosing between episodes in a podcast client
    sees the render's preset context (turn count, model) and what window of
    source material it actually covers, not just a bare date, since
    `list_watchlist_audio_episodes` carries no separate episode title or
    summary text of its own. `covers_from_ts` (Task 4 review round 1: was
    present on the row but unused) answers "what period does this episode
    actually cover" -- the one question a title alone cannot.

    Args:
        row: One `list_watchlist_audio_episodes` row.

    Returns:
        A one-line human-readable description.
    """
    details: list[str] = []
    turn_count = row.get("turn_count")
    if turn_count:
        details.append(f"{turn_count} turns")
    model_used = row.get("model_used")
    if model_used:
        details.append(f"model: {model_used}")
    created_at = row.get("briefing_created_at") or "an unknown time"
    suffix = f" ({', '.join(details)})" if details else ""
    covers_from = row.get("covers_from_ts")
    coverage = f" Covers content since {covers_from}." if covers_from else ""
    return f"Briefing from {created_at}{suffix}.{coverage}"


def _published_from_briefing_created_at(value: Any) -> datetime:
    """A `briefings.created_at` SQLite timestamp as an aware UTC `datetime`.

    SQLite's `CURRENT_TIMESTAMP` default has no explicit timezone, and this
    database's convention is that it is always UTC -- see `briefing_feed`'s
    module docstring, "All timestamps must be timezone-aware". A naive
    value passed straight through would make `build_feed_xml` raise on
    purpose rather than silently mis-schedule playback, so `timezone.utc`
    is attached explicitly, once, here.

    Args:
        value: `row["briefing_created_at"]`, expected in SQLite's
            `"YYYY-MM-DD HH:MM:SS"` form.

    Returns:
        The parsed timestamp, tagged `timezone.utc`.

    Raises:
        ValueError: `value` is not in the expected format. The caller
            treats this as one bad episode (see decision 3), not a fatal
            error for the whole export.
    """
    naive = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    return naive.replace(tzinfo=timezone.utc)


def _copy_episode_audio_file(
    source_path: Path, destination_dir: Path, filename: str
) -> Path:
    """Copy one episode's audio file into the feed directory.

    Follows the exact validate-then-copy order `UI/STTS_Window.py:4040-4049`
    uses for its own user-chosen export destination: validate the full
    destination path, validate its parent exists and resolve it (catches a
    symlink swapped in after the caller's own directory-level validation),
    validate the bare filename, THEN copy. `source_path` is not
    re-validated here -- by the time this is called, the caller has already
    confirmed it via `audio_file_path_is_safe` and `.exists()` (module
    docstring, decision 2).

    Task 4 review round 1: `shutil.copy2` preserves the SOURCE file's
    permission mode along with its data, and production audio is written
    by `Utils.private_paths.atomic_private_write_bytes` at `0o600`
    (application-owned, private-storage semantics). Left alone, every
    exported episode would inherit that private mode into the user's own
    folder -- exactly what Decision 1 forbids, just applied to a file
    instead of the directory, and defeating the entire point of exporting
    a folder the user means to sync, zip, or serve. The explicit `chmod`
    below relaxes it back to an ordinary, group/other-readable file
    regardless of what the source's mode happened to be.

    Args:
        source_path: The episode's audio file inside `briefing_audio_dir()`.
        destination_dir: The already-validated feed directory.
        filename: The bare filename to write within `destination_dir`.

    Returns:
        The validated path the file was copied to.

    Raises:
        ValueError: `destination_dir / filename` fails validation.
        OSError: The copy itself fails.
    """
    dest_path = destination_dir / filename
    validate_path_simple(dest_path, require_exists=False)
    validated_parent = validate_path_simple(
        dest_path.parent, require_exists=True
    ).resolve()
    validated_filename = validate_filename(dest_path.name)
    dest_path = validated_parent / validated_filename
    shutil.copy2(source_path, dest_path)
    os.chmod(dest_path, _EXPORTED_FILE_MODE)
    return dest_path


def _write_feed_xml_atomically(destination_dir: Path, xml_bytes: bytes) -> None:
    """Write `feed.xml` so a crash mid-write never corrupts a valid one.

    The `Chatbooks/chatbook_creator.py:1506` `_create_zip_archive` recipe,
    applied to a flat byte string instead of a zip stream: open the partial
    exclusively (`O_EXCL` -- never silently overwrite a concurrent writer;
    `O_NOFOLLOW` -- never follow a symlink planted at the partial's name),
    write, `flush()` + `fsync()` to force the bytes to disk, then
    `os.replace` -- atomic on every platform this app ships on -- to
    publish it under the real name. A stale partial left behind by a
    previous crash is removed first (best-effort): unlike
    `chatbook_creator`'s per-export unique output path, `feed.xml` is a
    fixed name reused on every re-export, so leaving a stale partial in
    place would permanently wedge every future export behind `O_EXCL`.

    Task 4 review round 1: the partial used to be opened at `0o600` (copied
    from the `chatbook_creator` precedent verbatim, without noticing that
    precedent's own file is APPLICATION-owned, unlike this one) -- and
    `os.replace` preserves the replaced-in file's mode, so every exported
    `feed.xml` inherited that private mode into the user's folder. Opened
    at `_EXPORTED_FILE_MODE` (`0o644`) instead, with an explicit `fchmod`
    right after creation so the result is deterministic regardless of the
    calling process's umask -- the same defensive-`fchmod` idiom
    `chatbook_creator._create_zip_archive` itself uses for its own mode.

    Args:
        destination_dir: The validated feed directory (already confirmed to
            exist by the caller).
        xml_bytes: The complete RSS document from `build_feed_xml`.

    Raises:
        OSError: The write or the final `os.replace` fails. Unlike a single
            episode's copy failure, this is NOT swallowed into `skipped` --
            a feed directory with no `feed.xml` is not a partial export, it
            is not an export at all (module docstring, decision 3's carve-out).
    """
    final_path = destination_dir / _FEED_XML_NAME
    partial_path = destination_dir / _FEED_XML_PARTIAL_NAME

    try:
        if partial_path.exists():
            partial_path.unlink()
    except OSError as exc:
        logger.warning(
            "Feed export: could not remove a stale partial feed ({}).",
            type(exc).__name__,
        )

    file_fd = -1
    partial_created = False
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(partial_path, flags, _EXPORTED_FILE_MODE)
        partial_created = True
        if hasattr(os, "fchmod"):
            os.fchmod(file_fd, _EXPORTED_FILE_MODE)
        with os.fdopen(file_fd, "wb") as stream:
            file_fd = -1  # ownership transferred to the stream
            stream.write(xml_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial_path, final_path)
        partial_created = False
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if partial_created:
            try:
                if partial_path.is_file():
                    partial_path.unlink()
            except OSError as exc:
                logger.warning(
                    "Feed export: could not remove the partial feed after a "
                    "failed write ({}).",
                    type(exc).__name__,
                )


def export_feed_directory(
    db: Any,
    watchlist_id: int,
    *,
    destination: Path,
    watchlist_name: str,
    now: datetime,
) -> FeedExportResult:
    """Write a watchlist's audio episodes as a self-contained podcast feed.

    The deliverable this whole spec exists to produce: a folder containing
    `feed.xml` plus a copy of every playable episode's audio, ready to hand
    to a podcast client, sync, or zip up as-is (`briefing_feed`'s module
    docstring -- enclosure URLs are bare relative filenames for exactly this
    reason).

    Per-episode failures (an unsafe `file_path`, a vanished source file, a
    copy that raises) are never fatal to the whole export -- each is
    recorded as a reason in the returned `FeedExportResult.skipped` and the
    remaining episodes still export (module docstring, decision 3). The
    safety check on each episode's `file_path` runs before ANY filesystem
    access on it (decision 2) -- no `Path(...)`, `.exists()`, `.stat()`, or
    `shutil.copy2` until `audio_file_path_is_safe` has passed.

    Args:
        db: An open `SubscriptionsDB`.
        watchlist_id: Passed straight to `list_watchlist_audio_episodes`.
        destination: The user-chosen directory to export into. Must already
            exist (a `SelectDirectory` dialog, Task 5, only ever dismisses
            with an existing directory or `None`) -- validated with
            `Utils.path_validation.validate_path_simple` before anything
            else runs, so a hostile or malformed path raises before the DB
            is even queried. NEVER routed through `Utils/private_paths.py`
            (module docstring, decision 1): this is the user's own folder,
            not application-owned storage.
        watchlist_name: Display name, used for the feed channel's title and
            description.
        now: Timestamp for the feed's `<lastBuildDate>`. Must be timezone-
            aware -- see `briefing_feed.build_feed_xml`.

    Returns:
        `FeedExportResult` naming the resolved directory, how many episodes
        made it into `feed.xml`, and a reason for each one that did not.

    Raises:
        ValueError: `destination` fails `validate_path_simple` (including
            not existing).
        FeedBuildError: `now` is naive (see `briefing_feed.build_feed_xml`).
        OSError: `feed.xml` itself could not be written atomically. Unlike
            a single episode's copy failure, this is fatal -- see
            `_write_feed_xml_atomically`.
    """
    validated_destination = validate_path_simple(
        destination, require_exists=True
    ).resolve()

    rows = db.list_watchlist_audio_episodes(watchlist_id)

    episodes: list[FeedEpisode] = []
    skipped: list[str] = []

    for row in rows:
        audio_id = row.get("audio_id")
        file_path = row.get("file_path")

        if not file_path:
            skipped.append(f"audio {audio_id}: no file path recorded")
            continue

        # Decision 2 (module docstring): the safety check is the FIRST
        # thing done with a DB-sourced file_path -- before Path(...),
        # .exists(), or any copy -- exactly the order phase 2b's Qodo round
        # established for playback (artifacts_pane.py's own
        # handle_play_audio_requested).
        if not audio_file_path_is_safe(file_path):
            skipped.append(f"audio {audio_id}: file path failed the safety check")
            continue

        source_path = Path(file_path)
        if not source_path.exists():
            skipped.append(f"audio {audio_id}: source file no longer exists")
            continue

        # Parsed before the copy so a malformed timestamp never leaves a
        # copied-but-never-listed orphan file behind in the user's
        # destination directory -- fail fast on bad metadata first.
        raw_created_at = row.get("briefing_created_at")
        try:
            published = _published_from_briefing_created_at(raw_created_at)
        except ValueError:
            skipped.append(
                f"audio {audio_id}: briefing_created_at {raw_created_at!r} "
                "is not a recognized timestamp"
            )
            continue

        try:
            filename = _episode_filename(row, source_path)
            dest_path = _copy_episode_audio_file(
                source_path, validated_destination, filename
            )
            length_bytes = dest_path.stat().st_size
        except OSError as exc:
            skipped.append(
                f"audio {audio_id}: could not copy the audio file "
                f"({type(exc).__name__})"
            )
            continue
        except ValueError as exc:
            skipped.append(f"audio {audio_id}: {exc}")
            continue

        episodes.append(
            FeedEpisode(
                title=_episode_title(row, published),
                filename=filename,
                length_bytes=length_bytes,
                duration_seconds=row.get("duration_seconds"),
                published=published,
                guid=f"briefing-audio-{audio_id}",
                description=_episode_description(row),
            )
        )

    xml_bytes = build_feed_xml(
        channel_title=f"{watchlist_name} — Audio Briefings",
        channel_description=(
            f'Audio briefings generated from the "{watchlist_name}" watchlist.'
        ),
        episodes=episodes,
        now=now,
    )
    _write_feed_xml_atomically(validated_destination, xml_bytes)

    return FeedExportResult(
        directory=validated_destination,
        episode_count=len(episodes),
        skipped=skipped,
    )
