"""Progressive, cancellable picker listings (ADR-160).

Records stay cheap; only visible rows or an explicit metadata sort need stat.
The original navigation/message contract is retained for all picker consumers.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from time import monotonic

from rich.table import Table
from rich.text import Text
from textual import work
from textual.message import Message
from textual.reactive import reactive
from textual.visual import RichVisual
from textual.widgets import OptionList
from textual.widgets.option_list import Option
from textual.worker import get_current_worker

from ....Utils.path_validation import validate_browsing_path
from .directory_navigation import (
    DirectoryEntry,
    DirectoryEntryStyling,
    _human_readable_size,
)
from .directory_navigation import (
    DirectoryNavigation as OriginalDirectoryNavigation,
)

SORT_OPTIONS = [
    # task-32611 AC#4 / task-32643 AC#1: folders first, name-ascending. It is
    # the DEFAULT only on a dialog that can return a folder (see
    # `FileSystemPickerScreen._default_listing_sort`); a file picker still
    # opens on "Discovery order", which stays on the menu for both.
    ("Folders first", "folders"),
    ("Discovery order", "discovery"),
    ("Name", "name"),
    ("Last modified", "modified"),
    ("Last accessed", "accessed"),
    ("Created", "created"),
    ("Size", "size"),
]

NOTE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
"""Extensions the Notes importer/sync treat as a note, for the folder badge."""

NOTE_COUNT_CEILING = 500
"""Hard bound on one folder's badge count (task-32643 AC#2).

Reached only by `count_folder_notes`, which reads ONE directory with a single
`os.scandir` and never descends. The ceiling bounds the pathological case (a
flat folder of 200k files) so a badge can never cost more than 500 `DirEntry`
reads; past it the badge reads "500+ notes" instead of a number.
"""

VAULT_MARKER = " · vault"
"""Suffix appended to an Obsidian vault's name in the listing (task-32643 AC#4)."""


def count_folder_notes(folder: Path, ceiling: int = NOTE_COUNT_CEILING) -> int | None:
    """Count note files sitting DIRECTLY inside one folder. Blocking.

    Deliberately depth-1 and capped: a picker that walks a subtree to draw a
    badge is worse than a picker with no badge, and this programme already has
    a P0 in its history from an abandoned recursive folder scan. One
    `os.scandir`, at most `ceiling` entries considered, no recursion, no stat
    (`DirEntry.is_file` uses the type the directory read already returned on
    every platform this app ships on).

    Args:
        folder: The directory to read.
        ceiling: Stop counting here; the caller renders the cap with a "+".

    Returns:
        The number of note files found, capped at `ceiling`, or None when the
        folder cannot be read (permissions, a race, a dead symlink).
    """
    total = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                if os.path.splitext(entry.name)[1].casefold() in NOTE_SUFFIXES:
                    total += 1
                    if total >= ceiling:
                        return ceiling
    except OSError:
        return None
    return total


@dataclass(frozen=True)
class FileRecord:
    """One directory entry, optionally hydrated with filesystem metadata."""

    location: Path
    is_directory: bool
    is_symlink: bool = False
    metadata: os.stat_result | None = None
    metadata_loaded: bool = False
    note_count: int | None = None
    is_vault: bool = False
    folder_summary_loaded: bool = False

    @property
    def display_name(self) -> str:
        """Keep unusual filesystem names on one terminal row."""
        name = (
            self.location.name.replace("\n", "⏎").replace("\r", "␍").replace("\t", "⇥")
        )
        return f"{name}{VAULT_MARKER}" if self.is_vault else name

    @property
    def size_text(self) -> str:
        if self.is_directory:
            # task-32643 AC#2: the note badge reuses the size column, which a
            # directory has always left blank -- no new column, so both the
            # vendored row and `EnhancedFileDialog`'s responsive one pick it
            # up unchanged. Blank until the bounded count lands, and blank
            # forever on a picker that did not ask for one.
            if self.note_count is None:
                return ""
            if self.note_count >= NOTE_COUNT_CEILING:
                return f"{NOTE_COUNT_CEILING}+ notes"
            return f"{self.note_count} note" + ("" if self.note_count == 1 else "s")
        return _human_readable_size(self.metadata.st_size) if self.metadata else "—"

    @property
    def modified_text(self) -> str:
        if self.metadata is None:
            return "—"
        try:
            return (
                datetime.fromtimestamp(self.metadata.st_mtime, tz=UTC)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
            )
        except (ValueError, OverflowError, OSError):
            return "—"


def read_metadata(record: FileRecord) -> FileRecord:
    """Read metadata in a worker, retaining missing or inaccessible entries.

    Args:
        record: Snapshot to hydrate; an already loaded snapshot is reused.

    Returns:
        A snapshot marked loaded, with stat metadata or None after an OSError.
    """
    if record.metadata_loaded:
        return record
    try:
        metadata = record.location.stat()
    except OSError:
        metadata = None
    return replace(record, metadata=metadata, metadata_loaded=True)


def read_folder_summary(record: FileRecord) -> FileRecord:
    """Hydrate one VISIBLE folder's note badge and vault marker. Blocking.

    Both halves read only the folder itself: `count_folder_notes` is one
    bounded `os.scandir`, and the vault question goes to the ONE predicate the
    rest of the app already uses (`Notes.note_import_discovery.
    folder_is_obsidian_vault`, the same `.obsidian/` marker Import once and
    "Keep a folder synced" look for) rather than a second one written here.

    Imported lazily: this vendored package is deliberately absent from the
    app's boot import closure (`Tests/Packaging/test_app_import_diet_closure`),
    and so is `note_import_discovery` -- a module-level import here would put
    the Notes discovery module on every picker's import path for a predicate
    only three of them ask for.

    Args:
        record: Snapshot to hydrate; a non-folder or already-summarised
            snapshot is returned unchanged except for the "done" flag.

    Returns:
        A snapshot marked `folder_summary_loaded`, so an unreadable folder is
        asked once and not re-queued on every scroll.
    """
    if record.folder_summary_loaded or not record.is_directory:
        return replace(record, folder_summary_loaded=True)
    from ....Notes.note_import_discovery import folder_is_obsidian_vault

    return replace(
        record,
        note_count=count_folder_notes(record.location),
        is_vault=folder_is_obsidian_vault(record.location),
        folder_summary_loaded=True,
    )


def project_records(
    records: Sequence[FileRecord],
    *,
    show_hidden: bool,
    query: str,
    file_filter: Callable[[Path], bool] | None,
    sort_key: str,
    descending: bool,
    cancelled: Event,
) -> tuple[list[FileRecord], int]:
    """Filter and sort snapshots off-loop, preserving unknown timestamps last.

    Args:
        records: Discovered entries in filesystem order.
        show_hidden: Whether dot-prefixed entries are visible.
        query: Stripped, casefolded filename substring to match.
        file_filter: Optional caller predicate, applied to non-directories.
        sort_key: Folders-first, discovery, name, modified, accessed, created,
            or size ordering.
        descending: Whether known sort values are ordered descending.
        cancelled: Cooperative cancellation signal checked between entries.

    Returns:
        Visible records and the count excluded by the file filter. Metadata
        sorts hydrate matching records; unreadable metadata remains unknown.
        Cancellation returns an empty projection which the owner must discard.

    Raises:
        Exception: A caller-supplied file filter failed; handled by the worker.
    """
    visible = []
    hidden = 0
    for record in records:
        if cancelled.is_set():
            return [], hidden
        dot_hidden = record.location.name.startswith(".") and not show_hidden
        if (
            file_filter is not None
            and not record.is_directory
            and not file_filter(record.location)
        ):
            hidden += not dot_hidden
            continue
        if dot_hidden or (query and query not in record.location.name.casefold()):
            continue
        if sort_key not in ("discovery", "name", "folders"):
            record = read_metadata(record)
        visible.append(record)
    if sort_key in ("name", "folders"):
        visible.sort(
            key=lambda r: (r.location.name.casefold(), r.location.name),
            reverse=descending,
        )
        if sort_key == "folders":
            # A SECOND, stable sort on the one bit that separates the groups
            # (task-32611 AC#4). Folding `not is_directory` into the key above
            # instead would make "Descending" put files first; here descending
            # reverses the NAMES and leaves folders on top, which is what
            # "Folders first" has to keep meaning in both directions.
            visible.sort(key=lambda r: not r.is_directory)
    elif sort_key != "discovery":
        attribute = {
            "modified": "st_mtime",
            "accessed": "st_atime",
            "created": "st_birthtime",
            "size": "st_size",
        }[sort_key]
        known, unknown = [], []
        for record in visible:
            value = getattr(record.metadata, attribute, None)
            (unknown if value is None else known).append(record)
        known.sort(
            key=lambda r: (getattr(r.metadata, attribute), r.location.name.casefold()),
            reverse=descending,
        )
        visible = known + unknown
    return visible, hidden


class SnapshotDirectoryEntry(DirectoryEntry):
    """A compatible Option whose prompt never performs filesystem operations."""

    def __init__(self, record: FileRecord, styles: DirectoryEntryStyling) -> None:
        self.record = record
        self.location = record.location
        self._styles = styles
        Option.__init__(self, self._as_renderable(self.location))

    def _as_renderable(self, location):
        record = self.record
        table = Table.grid(expand=True)
        column_styles = {
            2: self._styles.name,
            3: self._styles.size,
            4: self._styles.time,
        }
        for column, width in enumerate((1, 3, None, 10, 20, 1)):
            table.add_column(
                width=width,
                ratio=1 if width is None else None,
                no_wrap=True,
                overflow="ellipsis",
                justify="right" if column in (3, 4) else "left",
                style=self._style(column_styles[column], location)
                if column in column_styles
                else None,
            )
        table.add_row(
            "",
            self.FOLDER_ICON if record.is_directory else self.FILE_ICON,
            Text.assemble(
                record.display_name, " ", self.LINK_ICON if record.is_symlink else ""
            ),
            record.size_text,
            record.modified_text,
            "",
        )
        return table


class _SingleLineVisual(RichVisual):
    """Measure a fixed-height row without rendering its offscreen Rich table."""

    def get_height(self, rules, width):
        return 1

    def get_optimal_width(self, rules, container_width):
        return container_width


class ProgressiveDirectoryNavigation(OriginalDirectoryNavigation):
    """Shared navigation with bounded publication and ephemeral snapshots."""

    search_filter = reactive("")
    sort_key = reactive("discovery")
    sort_descending = reactive(False)
    BATCH_SIZE = 64
    PUBLISH_INTERVAL = 0.03

    show_folder_notes = False
    """Draw a note count and a vault marker on folder rows (task-32643 AC#2/#4).

    Off for every picker in the app except the three Notes folder doors, which
    turn it on through `FileSystemPickerScreen`'s `notes_context`: "12 notes"
    is a useful badge when choosing where notes live and noise in a model-file
    or character-card picker. The work it adds is the same shape as the
    metadata hydration it rides along with -- VISIBLE rows only, one bounded
    `os.scandir` per folder, never re-asked once answered.
    """

    class ListingChanged(Message):
        def __init__(self, navigation: ProgressiveDirectoryNavigation) -> None:
            self.navigation = navigation
            super().__init__()

    def __init__(self, location: Path | str = ".") -> None:
        self._generation = 0
        self._revision = 0
        self._scan_cancel = Event()
        self._projection_cancel = Event()
        self._records = []
        self._metadata = {}
        self._display_records = []
        self._scan_queue = Queue(maxsize=4)
        self._scan_finished = False
        self._projection_running = False
        self._projection_owner = None
        self._projection_worker = None
        self._projection_dirty = False
        self._metadata_running = False
        self._listing_error = ""
        self._filter_hidden = 0
        self._listing_timer = None
        self._interaction_serial = 0
        self._restore_highlight: tuple[Path, int] | None = None
        self._projection_error = ""
        super().__init__(location)

    def _get_visual(self, option):
        if isinstance(option, DirectoryEntry):
            if option._visual is None:
                option._visual = _SingleLineVisual(self, option.prompt)
            return option._visual
        return super()._get_visual(option)

    def _get_dispatch_methods(self, method_name, message):
        if (
            isinstance(message, OptionList.OptionMessage)
            and message.option_list is self
            and message.option not in self._option_to_index
        ):
            message.stop()
            return
        for cls, method in super()._get_dispatch_methods(method_name, message):
            # The framework uses an unchecked index from the painted frame.
            # Navigation can replace that list before a queued click arrives.
            if method.__func__ is OptionList._on_click:
                continue
            yield cls, method

    def _get_option_render(self, option, style):
        return [
            strip.apply_meta({"picker_row": id(option)})
            for strip in super()._get_option_render(option, style)
        ]

    def _current_clicked_index(self, event):
        index = event.style.meta.get("option")
        if index is None or not 0 <= index < self.option_count:
            return None
        if event.style.meta.get("picker_row") != id(self.options[index]):
            return None
        return index

    def _on_click(self, event):
        # Enhanced navigation handles the second click by opening the entry;
        # its handler runs before this one and may already have navigated.
        if event.chain > 1:
            return
        index = self._current_clicked_index(event)
        if (
            index is not None
            and 0 <= index < self.option_count
            and not self.options[index].disabled
        ):
            self._interaction_serial += 1
            self.highlighted = index
            self.action_select()

    def on_key(self, event) -> None:
        # Activation does not choose a new row. Invalidating restoration for
        # Enter lets a publication batch highlight '..' before its binding runs.
        if event.key in ("up", "down", "home", "end", "pageup", "pagedown"):
            self._interaction_serial += 1

    def _settle_highlight(self) -> None:
        if self.highlighted is not None:
            return
        first_entry = 0 if self.is_root else 1
        if self.option_count > first_entry:
            self.highlighted = first_entry
        elif self._scan_finished and self.option_count:
            self.highlighted = 0

    @property
    def listing_status(self) -> str:
        count = len(self._records)
        if error := self._listing_error or self._projection_error:
            return f"Stopped · {count} entries · {error}"
        if not self._scan_finished:
            return f"Scanning… · {count} entries found"
        if self._projection_running or self._projection_dirty:
            return f"{'Sorting' if self.sort_key != 'discovery' else 'Displaying'}… · {count} entries"
        suffix = (
            " · unavailable creation times last" if self.sort_key == "created" else ""
        )
        return f"Loaded · {count} entries{suffix}"

    def _notify_listing(self):
        self.post_message(self.ListingChanged(self))

    def _load(self):
        # The first read initializes a reactive and can reenter this method.
        # Read inputs BEFORE assigning a generation/queue, so an outer load
        # supersedes the inner load instead of starting two scans on its queue.
        location, show_files = self.location, self.show_files
        self._stop_projection()
        self._metadata_running = False
        self._scan_cancel.set()
        self._projection_cancel.set()
        self._generation += 1
        self._revision += 1
        self._restore_highlight = None
        self._scan_cancel = Event()
        self._projection_cancel = Event()
        self._records = []
        self._metadata = {}
        self._display_records = []
        self._entries = []
        self._scan_queue = Queue(maxsize=4)
        self._scan_finished = False
        self._listing_error = ""
        self._projection_error = ""
        self._entry_styles = self._styles
        self.clear_options()
        if not self.is_root:
            self.add_option(self._make_entry(FileRecord(self.location / "..", True)))
        self._settle_highlight()
        if self._listing_timer is None:
            self._listing_timer = self.set_interval(
                self.PUBLISH_INTERVAL, self._poll_listing
            )
        self._scan(location, show_files, self._scan_queue, self._scan_cancel)
        self._request_projection()
        self._notify_listing()

    @work(thread=True, exclusive=True, group="picker-scan")
    def _scan(self, location, show_files, queue, cancelled):
        worker = get_current_worker()

        def publish(value):
            while not cancelled.is_set() and not worker.is_cancelled:
                try:
                    queue.put(value, timeout=self.PUBLISH_INTERVAL)
                    return True
                except Full:
                    pass
            return False

        batch = []
        last_publish = 0.0
        error = None
        try:
            location = validate_browsing_path(location)
            with os.scandir(location) as entries:
                for entry in entries:
                    if cancelled.is_set() or worker.is_cancelled:
                        return
                    try:
                        try:
                            directory = entry.is_dir()
                            include = directory or (show_files and entry.is_file())
                        except PermissionError:
                            directory, include = False, show_files
                        if include:
                            try:
                                symlink = entry.is_symlink()
                            except PermissionError:
                                symlink = False
                            batch.append(
                                FileRecord(location / entry.name, directory, symlink)
                            )
                    except OSError:
                        continue
                    now = monotonic()
                    if batch and (
                        len(batch) >= self.BATCH_SIZE
                        or now - last_publish >= self.PUBLISH_INTERVAL
                    ):
                        if not publish((batch, False, None)):
                            return
                        batch = []
                        last_publish = now
        except (OSError, ValueError) as exc:
            error = exc
        publish((batch, True, error))

    def _poll_listing(self):
        changed = False
        # Drain at most the bounded queue capacity each tick, not one item:
        # a fast local disk should not be throttled to 64 files per frame.
        for _ in range(self._scan_queue.maxsize):
            try:
                batch, done, error = self._scan_queue.get_nowait()
            except Empty:
                break
            self._records.extend(batch)
            self._scan_finished = done
            self._listing_error = (
                f"Cannot read folder ({type(error).__name__})" if error else ""
            )
            if isinstance(error, PermissionError):
                self.post_message(self.PermissionError(self, self.location))
            changed = True
            if done:
                break
        if changed:
            self._request_projection()
            self._notify_listing()
        if not self._metadata_running:
            start = max(0, int(self.scroll_y) - 8)
            stop = min(self.option_count, int(self.scroll_y) + self.size.height + 8)
            records = []
            for i in range(start, stop):
                option = self.options[i]
                if getattr(option, "record", None) is None:
                    continue
                if option.location.name == "..":
                    continue
                record = self._metadata.get(option.location, option.record)
                if self._wants_hydration(record):
                    records.append(record)
            if records:
                self._metadata_running = True
                self._hydrate_visible(self._generation, records)

    def _wants_hydration(self, record) -> bool:
        """Is there still bounded per-row work owed on this visible record?

        The folder badge is a SECOND reason a row can be unhydrated, and it has
        to be asked independently of `metadata_loaded`: a metadata sort
        (`project_records`) stats every record off-loop without ever counting
        notes, so keying the queue on `metadata_loaded` alone left every folder
        that first became visible under "Size" or "Last modified" permanently
        badge-less.
        """
        if not record.metadata_loaded:
            return True
        return (
            self.show_folder_notes
            and record.is_directory
            and not record.folder_summary_loaded
        )

    def _repopulate_display(self):
        self._stop_projection()
        self._revision += 1
        self._projection_cancel.set()
        self._projection_cancel = Event()
        self._request_projection()

    def _watch_sort_key(self):
        self._repopulate_display()

    def _watch_sort_descending(self):
        self._repopulate_display()

    def _request_projection(self):
        self._projection_dirty = True
        if self.is_mounted and not self._projection_running:
            self._projection_running = True
            self._projection_owner = object()
            self._projection_worker = self._project(self._projection_owner)

    def _stop_projection(self):
        self._projection_owner = None
        if self._projection_worker is not None:
            self._projection_worker.cancel()
        self._projection_running = False

    def _make_entry(self, record):
        return SnapshotDirectoryEntry(record, self._entry_styles)

    def _replace_record(self, index, record):
        record = self._metadata.get(record.location, record)
        option = self.options[index]
        if option.record != record:
            option.record = record
            self.replace_option_prompt_at_index(index, self._make_entry(record).prompt)

    def _settle_projection_highlight(self, *, final=False):
        if self._apply_pending_highlight(
            final=final and self._scan_finished and not self._projection_dirty
        ):
            self._restore_highlight = None
            return
        if self._restore_highlight is not None:
            path, interaction = self._restore_highlight
            if interaction == self._interaction_serial:
                # Keep the restore target across cancelled rebuilds. While
                # publishing, no fallback row may silently become actionable.
                if not final:
                    return
                for index, option in enumerate(self.options):
                    if option.location == path:
                        self.highlighted = index
                        break
            self._restore_highlight = None
        self._settle_highlight()

    @work(exclusive=True, group="picker-projection")
    async def _project(self, owner):
        try:
            while self._projection_owner is owner and self._projection_dirty:
                self._projection_dirty = False
                generation, revision = self._generation, self._revision
                cancelled = self._projection_cancel
                records = [self._metadata.get(r.location, r) for r in self._records]
                query = self.search_filter.strip().casefold()
                try:
                    visible, hidden = await asyncio.to_thread(
                        project_records,
                        records,
                        show_hidden=self.show_hidden,
                        query=query,
                        file_filter=self.file_filter,
                        sort_key=self.sort_key,
                        descending=self.sort_descending,
                        cancelled=cancelled,
                    )
                except Exception as exc:  # noqa: BLE001 -- user-supplied filter boundary
                    if generation == self._generation and revision == self._revision:
                        self._projection_error = (
                            f"Cannot filter/sort ({type(exc).__name__})"
                        )
                    continue
                if generation != self._generation or revision != self._revision:
                    continue
                self._projection_error = ""
                self._entry_styles = self._styles
                self._metadata.update(
                    (r.location, r) for r in visible if r.metadata_loaded
                )
                old = self._display_records
                append = len(old) <= len(visible) and all(
                    a.location == b.location for a, b in zip(old, visible)
                )
                offset = len(old) if append else 0
                retained = (
                    {}
                    if append
                    else {option.location: option for option in self.options}
                )

                def make_entry(record, retained=retained):
                    record = self._metadata.get(record.location, record)
                    option = retained.get(record.location)
                    if option is not None and option.record == record:
                        return option
                    return self._make_entry(record)

                if not append:
                    if (
                        self._restore_highlight is None
                        or self._restore_highlight[1] != self._interaction_serial
                    ):
                        previous = self.highlighted_option
                        self._restore_highlight = (
                            (previous.location, self._interaction_serial)
                            if isinstance(previous, DirectoryEntry)
                            else None
                        )
                    self.clear_options()
                    self._display_records = []
                    if not self.is_root:
                        self.add_option(
                            self._make_entry(FileRecord(self.location / "..", True))
                        )
                else:
                    parent_offset = 0 if self.is_root else 1
                    for start in range(0, offset, self.BATCH_SIZE):
                        batch = visible[start : min(start + self.BATCH_SIZE, offset)]
                        for index, record in enumerate(batch, start + parent_offset):
                            self._replace_record(index, record)
                        self._display_records[start : start + len(batch)] = batch
                        await asyncio.sleep(0)
                for start in range(offset, len(visible), self.BATCH_SIZE):
                    if generation != self._generation or revision != self._revision:
                        break
                    batch = visible[start : start + self.BATCH_SIZE]
                    self.add_options(make_entry(r) for r in batch)
                    self._display_records.extend(batch)
                    self._settle_projection_highlight()
                    await asyncio.sleep(0)
                else:
                    self._filter_hidden = hidden
                    self._settle_projection_highlight(final=True)
                    self._notify_listing()
        finally:
            if self._projection_owner is owner:
                self._projection_running = False
                self._notify_listing()

    @work(exclusive=True, group="picker-visible-metadata")
    async def _hydrate_visible(self, generation, records):
        try:
            cancelled = self._scan_cancel
            summarise_folders = self.show_folder_notes

            def hydrate():
                hydrated = []
                for record in records:
                    if cancelled.is_set():
                        break
                    record = read_metadata(record)
                    if summarise_folders and record.is_directory:
                        record = read_folder_summary(record)
                    hydrated.append(record)
                return hydrated

            hydrated = await asyncio.to_thread(hydrate)
            if generation != self._generation:
                return
            self._metadata.update((r.location, r) for r in hydrated)
            replacements = {r.location: r for r in hydrated}
            for index, option in enumerate(self.options):
                if option.location in replacements:
                    record = replacements[option.location]
                    self._replace_record(index, record)
        finally:
            if generation == self._generation:
                self._metadata_running = False

    def on_unmount(self):
        self._scan_cancel.set()
        self._projection_cancel.set()
        self._generation += 1
        if self._listing_timer is not None:
            self._listing_timer.stop()
