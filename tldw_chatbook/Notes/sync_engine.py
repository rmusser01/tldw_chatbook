# sync_engine.py
# Description: Engine for bi-directional file synchronization of notes
#
# Imports
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field

#
# Third-Party Imports
from loguru import logger

#
# Local Imports
from ..DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from .Notes_Library import NotesInteropService
from ..Metrics.metrics_logger import log_counter, log_histogram
from .sync_paths import PinnedSyncRoot, SafeSyncFile, SyncPathError
#
########################################################################################################################
#
# Classes and Functions:


# --- Conflict preservation (task-19554) ----------------------------------
#
# A resolution that overwrites one side writes that side's text, verbatim,
# to a sidecar next to the note file BEFORE the overwrite happens, so
# recovery is a rename. The marker is what makes a sidecar recognizable:
# ``_scan_directory`` drops anything carrying it so a preserved copy can
# never be re-ingested as a note (which would turn every conflict into a
# duplicate note, and every subsequent pass into another one). The ``.bak``
# suffix is belt to that braces -- it is outside the default
# ``['.md', '.txt']`` scan set as well.
CONFLICT_SIDECAR_MARKER = ".conflict-"
CONFLICT_SIDECAR_SUFFIX = ".bak"
_CONFLICT_SIDECAR_ATTEMPTS = 64

#: The two sides a conflict has: used both for the winner a strategy
#: picks and for the side a resolution throws away.
SIDE_DB = "db"
SIDE_DISK = "disk"

#: Values ``sync_conflicts.resolution`` accepts (its CHECK constraint).
RESOLUTION_USE_DB = "use_db"
RESOLUTION_USE_DISK = "use_disk"


class SyncDirection(Enum):
    """Enumeration for sync directions."""

    DISK_TO_DB = "disk_to_db"
    DB_TO_DISK = "db_to_disk"
    BIDIRECTIONAL = "bidirectional"


class ConflictResolution(Enum):
    """Enumeration for conflict resolution strategies."""

    ASK = "ask"
    DISK_WINS = "disk_wins"
    DB_WINS = "db_wins"
    NEWER_WINS = "newer_wins"


class SyncStatus(Enum):
    """Enumeration for sync session status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SyncFileInfo:
    """Information about a file for syncing."""

    absolute_path: Path
    relative_path: Path
    content: str
    content_hash: str
    mtime: float
    extension: str


@dataclass
class SyncConflict:
    """Represents a sync conflict.

    The last four fields record what the run actually DID about it, which is
    not the same thing as the policy that was selected (task-19554: every
    strategy but ``NEWER_WINS`` used to apply nothing on a ``both_changed``
    conflict while the UI reported it as resolved).

    Attributes:
        applied: Whether this run changed a side because of this conflict.
            ``False`` means the conflict is still open -- never report it as
            resolved.
        resolution: The value stamped into ``sync_conflicts.resolution``:
            ``"use_db"``, ``"use_disk"``, or ``None`` when nothing was applied.
        preserved_path: Absolute path of the sidecar holding the discarded
            side, when one was discarded.
        row_id: ``sync_conflicts.id`` of the recorded row, so the outcome can
            be stamped back onto it.
    """

    note_id: Optional[str]
    file_path: Path
    conflict_type: str
    db_content: Optional[str] = None
    disk_content: Optional[str] = None
    db_hash: Optional[str] = None
    disk_hash: Optional[str] = None
    db_modified: Optional[datetime] = None
    disk_modified: Optional[float] = None
    applied: bool = False
    resolution: Optional[str] = None
    preserved_path: Optional[Path] = None
    row_id: Optional[int] = None


@dataclass
class SyncProgress:
    """Tracks sync operation progress."""

    total_files: int = 0
    processed_files: int = 0
    conflicts: List[SyncConflict] = field(default_factory=list)
    errors: List[Tuple[str, Exception]] = field(default_factory=list)
    created_notes: List[str] = field(default_factory=list)
    updated_notes: List[str] = field(default_factory=list)
    created_files: List[Path] = field(default_factory=list)
    updated_files: List[Path] = field(default_factory=list)
    skipped_items: List[Tuple[str, str]] = field(default_factory=list)  # (item, reason)
    # Sidecars holding a discarded side. Deliberately NOT in
    # ``created_files``: they are not synced notes and must never be counted
    # as changes the run made to the user's content.
    preserved_files: List[Path] = field(default_factory=list)

    @property
    def applied_conflicts(self) -> List[SyncConflict]:
        """Conflicts this run actually resolved by changing a side."""

        return [conflict for conflict in self.conflicts if conflict.applied]

    @property
    def unresolved_conflicts(self) -> List[SyncConflict]:
        """Conflicts this run recorded but left for the user to settle."""

        return [conflict for conflict in self.conflicts if not conflict.applied]


class NotesSyncEngine:
    """Engine for synchronizing notes between database and file system."""

    def __init__(
        self,
        notes_service: NotesInteropService,
        db: CharactersRAGDB,
        progress_callback: Optional[Callable[[SyncProgress], None]] = None,
    ):
        """
        Initialize the sync engine.

        Args:
            notes_service: The notes service for database operations
            db: Direct database access for sync-specific operations
            progress_callback: Optional callback for progress updates
        """
        self.notes_service = notes_service
        self.db = db
        self.progress_callback = progress_callback
        self._active_sessions: Dict[str, SyncProgress] = {}
        self._cancelled_sessions: Set[str] = set()

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _coerce_aware_datetime(value: Any) -> datetime:
        """Coerce a DB ``last_modified`` value to a tz-aware ``datetime``.

        The value is normally the raw TEXT column (an ISO-8601 string),
        but ``sqlite3.PARSE_DECLTYPES`` (set on every ``CharactersRAGDB``
        connection) can auto-convert a TIMESTAMP-declared column to a real
        ``datetime`` depending on the call path -- accepting both here
        avoids a ``TypeError`` from ``datetime.fromisoformat`` when a
        ``datetime`` is already given. A naive result (no tzinfo, from
        either path) is assumed UTC, matching this module's other
        timestamps, so comparisons against tz-aware values never raise.
        """
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        return (
            parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        )

    def _get_file_info(
        self, file_path: Path, root_path: Path
    ) -> Optional[SyncFileInfo]:
        """Get file information for syncing."""
        try:
            try:
                relative_path = file_path.relative_to(root_path)
            except ValueError:
                relative_path = file_path.resolve(strict=False).relative_to(
                    root_path.resolve(strict=True)
                )
            with PinnedSyncRoot(root_path) as pinned_root:
                safe_file = pinned_root.read_file(relative_path)
            return self._from_safe_file(safe_file)
        except (OSError, UnicodeError, ValueError) as exc:
            logger.warning(
                "Could not read sync file (error_type={})",
                type(exc).__name__,
            )
            return None

    def _from_safe_file(self, safe_file: SafeSyncFile) -> SyncFileInfo:
        """Convert a descriptor-verified file into the engine's sync shape."""

        return SyncFileInfo(
            absolute_path=safe_file.absolute_path,
            relative_path=safe_file.relative_path,
            content=safe_file.content,
            content_hash=self._calculate_hash(safe_file.content),
            mtime=safe_file.mtime,
            extension=safe_file.extension,
        )

    def _write_file_info(
        self,
        pinned_root: PinnedSyncRoot,
        relative_path: Path,
        content: str,
    ) -> SyncFileInfo:
        """Write through the pinned boundary and return refreshed metadata."""

        return self._from_safe_file(
            pinned_root.write_text(relative_path, content)
        )

    @staticmethod
    def _record_path_skip(
        progress: SyncProgress,
        relative_path: Path,
        exc: SyncPathError,
    ) -> None:
        """Record one fail-closed entry without retaining raw error text."""

        progress.skipped_items.append((str(relative_path), exc.reason))
        logger.warning("Skipped sync entry (reason={})", exc.reason)

    @staticmethod
    def _path_was_rejected(progress: SyncProgress, relative_path: Path) -> bool:
        """Distinguish a rejected path from one confirmed absent on disk."""

        for item, _reason in progress.skipped_items:
            rejected = Path(item)
            if rejected == Path("."):
                continue
            if rejected == relative_path or rejected in relative_path.parents:
                return True
        return False

    def _scan_directory(
        self,
        root_path: Path,
        extensions: List[str] = None,
        *,
        pinned_root: PinnedSyncRoot | None = None,
        progress: SyncProgress | None = None,
    ) -> Dict[Path, SyncFileInfo]:
        """
        Scan directory for files to sync.

        Args:
            root_path: Root directory to scan
            extensions: List of file extensions to include (e.g., ['.md', '.txt'])
                       If None, defaults to ['.md', '.txt']

        Returns:
            Dictionary mapping relative paths to file info
        """
        start_time = time.time()
        log_counter(
            "sync_engine_scan_directory_attempt",
            labels={"extensions_count": str(len(extensions) if extensions else 2)},
        )

        if extensions is None:
            extensions = [".md", ".txt"]

        owns_root = pinned_root is None
        selected_root = pinned_root or PinnedSyncRoot(root_path)
        if owns_root:
            selected_root.__enter__()
        try:
            safe_files, issues = selected_root.scan(extensions)
        finally:
            if owns_root:
                selected_root.__exit__(None, None, None)
        # Preserved conflict copies are never sync candidates. Their ``.bak``
        # suffix already keeps them out of the default extension set, but the
        # marker check is what holds if a caller passes custom extensions:
        # ingesting one would create a duplicate note per conflict, and its
        # own file on the next pass.
        files_map = {
            relative_path: self._from_safe_file(safe_file)
            for relative_path, safe_file in safe_files.items()
            if not self.is_conflict_sidecar(relative_path)
        }
        files_failed = len(issues)
        if progress is not None:
            progress.skipped_items.extend(
                (str(issue.relative_path), issue.reason) for issue in issues
            )

        # Log metrics
        duration = time.time() - start_time
        log_histogram("sync_engine_scan_directory_duration", duration)
        log_histogram("sync_engine_scan_files_found", len(files_map))
        log_histogram("sync_engine_scan_files_failed", files_failed)
        log_counter(
            "sync_engine_scan_directory_success",
            labels={
                "files_found": str(len(files_map)),
                "files_failed": str(files_failed),
            },
        )

        logger.info(
            "Scanned sync root (files_found={}, files_skipped={})",
            len(files_map),
            files_failed,
        )
        return files_map

    def _get_synced_notes_for_root(
        self,
        root_path: Path,
        user_id: str,
        *,
        lexical_root: Path | None = None,
        progress: SyncProgress | None = None,
    ) -> Dict[Path, Dict[str, Any]]:
        """Get all notes that are synced to the given root folder."""
        db_notes_map = {}
        root_aliases = list(dict.fromkeys([str(root_path), str(lexical_root or root_path)]))
        placeholders = ", ".join("?" for _ in root_aliases)

        with self.db.transaction() as conn:
            cursor = conn.execute(
                f"""
                SELECT id, title, content, version, relative_file_path_on_disk,
                       last_synced_disk_file_hash, last_synced_disk_file_mtime,
                       last_modified, file_extension, sync_strategy, sync_excluded
                FROM notes
                WHERE deleted = 0 
                  AND sync_root_folder IN ({placeholders})
                  AND is_externally_synced = 1
                  AND sync_excluded = 0
            """,
                root_aliases,
            )

            for row in cursor:
                note_data = {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "version": row[3],
                    "relative_file_path_on_disk": row[4],
                    "last_synced_disk_file_hash": row[5],
                    "last_synced_disk_file_mtime": row[6],
                    "last_modified": row[7],
                    "file_extension": row[8] or ".md",
                    "sync_strategy": row[9],
                    "sync_excluded": row[10],
                }

                if note_data["relative_file_path_on_disk"]:
                    try:
                        rel_path = PinnedSyncRoot.validate_relative(
                            note_data["relative_file_path_on_disk"]
                        )
                    except SyncPathError as exc:
                        if progress is not None:
                            self._record_path_skip(
                                progress,
                                Path(note_data["relative_file_path_on_disk"]),
                                exc,
                            )
                        continue
                    note_data["content_hash"] = self._calculate_hash(
                        note_data["content"]
                    )
                    db_notes_map[rel_path] = note_data

        logger.info(
            "Found synced notes for selected root (count={})",
            len(db_notes_map),
        )
        return db_notes_map

    def _create_sync_session(
        self,
        sync_root: Path,
        direction: SyncDirection,
        conflict_resolution: ConflictResolution,
        user_id: str,
    ) -> str:
        """Create a new sync session in the database."""
        session_id = str(uuid.uuid4())

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO sync_sessions 
                (session_id, sync_root_folder, sync_direction, conflict_resolution, 
                 status, client_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    str(sync_root),
                    direction.value,
                    conflict_resolution.value,
                    SyncStatus.RUNNING.value,
                    user_id,
                ),
            )

        self._active_sessions[session_id] = SyncProgress()
        return session_id

    def _update_sync_session(
        self,
        session_id: str,
        progress: SyncProgress,
        status: SyncStatus,
        summary: Optional[Dict] = None,
    ):
        """Update sync session in the database."""
        with self.db.transaction() as conn:
            update_data = {
                "processed_files": progress.processed_files,
                "conflicts_found": len(progress.conflicts),
                "errors_count": len(progress.errors),
                "status": status.value,
            }

            if status in (
                SyncStatus.COMPLETED,
                SyncStatus.FAILED,
                SyncStatus.CANCELLED,
            ):
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

            if summary:
                update_data["summary"] = json.dumps(summary)

            # Build UPDATE query
            set_clauses = [f"{k} = ?" for k in update_data.keys()]
            values = list(update_data.values()) + [session_id]

            conn.execute(
                f"""
                UPDATE sync_sessions 
                SET {", ".join(set_clauses)}
                WHERE session_id = ?
            """,
                values,
            )

    def _record_conflict(self, session_id: str, conflict: SyncConflict) -> Optional[int]:
        """Record a sync conflict in the database.

        Only the detection facts are written here -- the discarded side's text
        is written later, by ``_resolve_with_preservation``, and only if a side
        is actually discarded. Recording content on mere DETECTION would make
        this table a second unbounded full-content shadow of ``notes``
        (``sync_log`` already is one) and would store a "backup" of text
        nothing was going to destroy.

        Args:
            session_id: The sync session recording the conflict.
            conflict: The conflict; its ``row_id`` is set from the new row.

        Returns:
            The new ``sync_conflicts.id``, or ``None`` if SQLite did not
            report one.
        """
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sync_conflicts
                (session_id, note_id, file_path, conflict_type,
                 db_content_hash, disk_content_hash,
                 db_modified_time, disk_modified_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    conflict.note_id,
                    str(conflict.file_path),
                    conflict.conflict_type,
                    conflict.db_hash,
                    conflict.disk_hash,
                    conflict.db_modified.isoformat() if conflict.db_modified else None,
                    conflict.disk_modified,
                ),
            )
            conflict.row_id = cursor.lastrowid
        return conflict.row_id

    def _update_conflict_record(
        self,
        conflict: SyncConflict,
        *,
        losing_side: Optional[str] = None,
        losing_content: Optional[str] = None,
        preserved_path: Optional[Path] = None,
        resolution: Optional[str] = None,
    ) -> None:
        """Stamp a recorded conflict with what was preserved and/or applied.

        ``resolution`` is left NULL until a side is genuinely applied, which is
        what keeps ``NotesSyncService.resolve_conflict`` (``WHERE resolution IS
        NULL``) able to find the conflicts that are still open.

        Args:
            conflict: The conflict whose row to update (no-op without a
                ``row_id``).
            losing_side: ``"db"``/``"disk"``, the side being discarded.
            losing_content: That side's text, verbatim.
            preserved_path: Absolute path of the sidecar holding it.
            resolution: ``"use_db"``/``"use_disk"`` once applied.
        """
        if conflict.row_id is None:
            return
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE sync_conflicts
                   SET losing_side = ?,
                       losing_content = ?,
                       preserved_file_path = ?,
                       resolution = ?,
                       resolved_at = ?
                 WHERE id = ?
            """,
                (
                    losing_side,
                    losing_content,
                    str(preserved_path) if preserved_path is not None else None,
                    resolution,
                    (
                        datetime.now(timezone.utc).isoformat()
                        if resolution is not None
                        else None
                    ),
                    conflict.row_id,
                ),
            )

    @staticmethod
    def is_conflict_sidecar(relative_path: Path) -> bool:
        """Return whether a scanned entry is one of our preserved copies."""

        return CONFLICT_SIDECAR_MARKER in relative_path.name

    def _write_conflict_sidecar(
        self,
        pinned_root: PinnedSyncRoot,
        relative_path: Path,
        side: str,
        content: str,
    ) -> Path:
        """Write the discarded side next to the note file, byte-exact.

        Verbatim content and nothing else -- no header, no diff markers -- so
        recovering the lost text is a rename, not an edit. Which side it holds
        and when it was taken are carried by the NAME
        (``note.md.conflict-20260821T203015Z-disk.bak``), which also sorts
        directly beside the file it came from.

        The write goes through the same ``PinnedSyncRoot`` boundary as every
        other sync write, so a swapped root or a symlinked path fails closed
        here exactly as it does for a note file.

        Args:
            pinned_root: The entered sync root.
            relative_path: The note file's path relative to that root.
            side: ``"db"`` or ``"disk"``.
            content: The text being discarded.

        Returns:
            Absolute path of the sidecar just written.

        Raises:
            SyncPathError: If a free name cannot be found, or the write is
                refused by the boundary.
            OSError: Propagated from the write.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for attempt in range(_CONFLICT_SIDECAR_ATTEMPTS):
            # Second-resolution stamps can repeat within one run, and an
            # earlier preserved copy must never be clobbered by a later one.
            ordinal = "" if attempt == 0 else f"-{attempt + 1}"
            candidate = relative_path.with_name(
                f"{relative_path.name}{CONFLICT_SIDECAR_MARKER}"
                f"{stamp}{ordinal}-{side}{CONFLICT_SIDECAR_SUFFIX}"
            )
            if (pinned_root.canonical_root / candidate).exists():
                continue
            return pinned_root.write_text(candidate, content).absolute_path
        raise SyncPathError("conflict_sidecar_name_exhausted", relative_path)

    def _resolve_with_preservation(
        self,
        *,
        conflict: SyncConflict,
        progress: SyncProgress,
        pinned_root: PinnedSyncRoot,
        relative_path: Path,
        losing_side: str,
        losing_content: str,
        resolution: str,
        apply_winner: Callable[[], None],
    ) -> bool:
        """Save the losing side, then apply the winner. Fail closed.

        The ordering is the whole point: if the discarded text cannot be
        written somewhere recoverable, the overwrite does **not** happen. A
        run that cannot preserve reports an error and leaves both sides alone,
        which is always recoverable; the alternative -- destroying anyway --
        never is.

        Args:
            conflict: The recorded conflict, updated in place with the outcome.
            progress: The run's progress, for the sidecar and any error.
            pinned_root: The entered sync root.
            relative_path: The note file's path relative to that root.
            losing_side: ``"db"``/``"disk"``, the side about to be discarded.
            losing_content: That side's text.
            resolution: ``"use_db"``/``"use_disk"`` to stamp once applied.
            apply_winner: Performs the destructive write. May raise; the
                caller's handler records it and the conflict stays unresolved
                with its copy already saved.

        Returns:
            Whether the winner was applied.
        """
        try:
            preserved = self._write_conflict_sidecar(
                pinned_root, relative_path, losing_side, losing_content
            )
        except Exception as exc:
            logger.error(
                "Refusing to resolve conflict: the losing side could not be "
                "preserved (side={}, error_type={})",
                losing_side,
                type(exc).__name__,
            )
            log_counter(
                "sync_engine_conflict_preservation_failed",
                labels={"losing_side": losing_side},
            )
            progress.errors.append((str(relative_path), exc))
            return False

        conflict.preserved_path = preserved
        progress.preserved_files.append(preserved)
        # Persist the copy BEFORE the overwrite too, so an apply that dies
        # part-way still leaves the discarded text reconstructible from the
        # row (with a NULL resolution saying the run never finished).
        self._update_conflict_record(
            conflict,
            losing_side=losing_side,
            losing_content=losing_content,
            preserved_path=preserved,
        )

        apply_winner()

        conflict.applied = True
        conflict.resolution = resolution
        self._update_conflict_record(
            conflict,
            losing_side=losing_side,
            losing_content=losing_content,
            preserved_path=preserved,
            resolution=resolution,
        )
        log_counter(
            "sync_engine_conflict_resolved",
            labels={"resolution": resolution, "losing_side": losing_side},
        )
        return True

    def _resolve_both_changed(
        self,
        *,
        conflict: SyncConflict,
        conflict_resolution: ConflictResolution,
        progress: SyncProgress,
        pinned_root: PinnedSyncRoot,
        relative_path: Path,
        root_path: Path,
        user_id: str,
        db_note: Dict[str, Any],
        disk_file: SyncFileInfo,
        may_write_disk: bool,
        may_write_db: bool,
        include_title: bool = False,
    ) -> None:
        """Apply the selected strategy to a ``both_changed`` conflict.

        One implementation for all three directions; the direction only
        restricts which side may be written (``may_write_*``). A strategy whose
        winner cannot be written in this direction applies nothing and leaves
        the conflict OPEN rather than pretending to have settled it -- e.g.
        "Disk wins" during a Library → Disk push has nothing to do, because the
        disk copy already stands.

        Args:
            conflict: The recorded conflict, updated in place.
            conflict_resolution: The selected strategy.
            progress: The run's progress.
            pinned_root: The entered sync root.
            relative_path: The note file's path relative to that root.
            root_path: The canonical sync root.
            user_id: User whose notes are being synced.
            db_note: The note row (content, version, id, last_modified).
            disk_file: The scanned file.
            may_write_disk: Whether this direction may overwrite the file.
            may_write_db: Whether this direction may overwrite the note.
            include_title: Whether applying the disk side also renames the note
                from the file stem (the disk → DB direction does; the
                bidirectional path never did).
        """
        winner = self._both_changed_winner(
            conflict_resolution, db_note, disk_file
        )

        if winner == SIDE_DB and may_write_disk:

            def apply_db_content() -> None:
                new_file_info = self._write_file_info(
                    pinned_root, relative_path, db_note["content"]
                )
                self._update_note_sync_metadata(
                    db_note["id"],
                    new_file_info,
                    root_path,
                    user_id,
                    db_note["version"],
                )
                progress.updated_files.append(new_file_info.absolute_path)

            self._resolve_with_preservation(
                conflict=conflict,
                progress=progress,
                pinned_root=pinned_root,
                relative_path=relative_path,
                losing_side=SIDE_DISK,
                losing_content=disk_file.content,
                resolution=RESOLUTION_USE_DB,
                apply_winner=apply_db_content,
            )
            return

        if winner == SIDE_DISK and may_write_db:

            def apply_disk_content() -> None:
                update_data: Dict[str, Any] = {"content": disk_file.content}
                if include_title:
                    update_data["title"] = disk_file.absolute_path.stem
                success = self.notes_service.update_note(
                    user_id=user_id,
                    note_id=db_note["id"],
                    update_data=update_data,
                    expected_version=db_note["version"],
                )
                if not success:
                    raise ConflictError(
                        f"Version mismatch applying disk content to note "
                        f"{db_note['id']}"
                    )
                updated_note = self.notes_service.get_note_by_id(
                    user_id, db_note["id"]
                )
                if updated_note:
                    self._update_note_sync_metadata(
                        db_note["id"],
                        disk_file,
                        root_path,
                        user_id,
                        updated_note["version"],
                    )
                progress.updated_notes.append(db_note["id"])

            self._resolve_with_preservation(
                conflict=conflict,
                progress=progress,
                pinned_root=pinned_root,
                relative_path=relative_path,
                losing_side=SIDE_DB,
                losing_content=db_note["content"],
                resolution=RESOLUTION_USE_DISK,
                apply_winner=apply_disk_content,
            )
            return

        logger.info(
            "Conflict left unresolved (strategy={}, winner={}, "
            "may_write_disk={}, may_write_db={})",
            conflict_resolution.value,
            winner,
            may_write_disk,
            may_write_db,
        )

    def _both_changed_winner(
        self,
        conflict_resolution: ConflictResolution,
        db_note: Dict[str, Any],
        disk_file: SyncFileInfo,
    ) -> Optional[str]:
        """Return which side a strategy picks, or ``None`` for "don't touch".

        ``ASK`` is the only strategy with no winner: it records the conflict
        for a human and changes nothing. (Nothing in the shipped UI offers it
        -- see the Library sync panel -- which is exactly why every other
        strategy has to actually apply.)
        """
        if conflict_resolution == ConflictResolution.DB_WINS:
            return SIDE_DB
        if conflict_resolution == ConflictResolution.DISK_WINS:
            return SIDE_DISK
        if conflict_resolution == ConflictResolution.NEWER_WINS:
            db_modified = self._coerce_aware_datetime(db_note["last_modified"])
            disk_modified = datetime.fromtimestamp(disk_file.mtime, tz=timezone.utc)
            return SIDE_DB if db_modified > disk_modified else SIDE_DISK
        return None

    def _update_note_sync_metadata(
        self,
        note_id: str,
        file_info: SyncFileInfo,
        root_path: Path,
        user_id: str,
        version: int,
    ):
        """Update note's sync metadata after successful sync."""
        with self.db.transaction() as conn:
            now = datetime.now(timezone.utc).isoformat()
            new_version = version + 1

            cursor = conn.execute(
                """
                UPDATE notes
                SET file_path_on_disk = ?,
                    relative_file_path_on_disk = ?,
                    sync_root_folder = ?,
                    last_synced_disk_file_hash = ?,
                    last_synced_disk_file_mtime = ?,
                    is_externally_synced = 1,
                    file_extension = ?,
                    last_modified = ?,
                    version = ?
                WHERE id = ? AND version = ?
            """,
                (
                    str(file_info.absolute_path),
                    str(file_info.relative_path),
                    str(root_path),
                    file_info.content_hash,
                    file_info.mtime,
                    file_info.extension,
                    now,
                    new_version,
                    note_id,
                    version,
                ),
            )

            if cursor.rowcount == 0:
                raise ConflictError(
                    f"Version mismatch updating sync metadata for note {note_id}"
                )

    def _unlink_note_from_sync(self, note_id: str, version: int):
        """Remove sync metadata from a note."""
        with self.db.transaction() as conn:
            now = datetime.now(timezone.utc).isoformat()
            new_version = version + 1

            cursor = conn.execute(
                """
                UPDATE notes
                SET file_path_on_disk = NULL,
                    relative_file_path_on_disk = NULL,
                    last_synced_disk_file_hash = NULL,
                    last_synced_disk_file_mtime = NULL,
                    is_externally_synced = 0,
                    last_modified = ?,
                    version = ?
                WHERE id = ? AND version = ?
            """,
                (now, new_version, note_id, version),
            )

            if cursor.rowcount == 0:
                raise ConflictError(f"Version mismatch unlinking note {note_id}")

    def cancel_sync(self, session_id: str):
        """Cancel an active sync session."""
        self._cancelled_sessions.add(session_id)
        logger.info(f"Sync session {session_id} marked for cancellation")

    def is_cancelled(self, session_id: str) -> bool:
        """Check if a sync session has been cancelled."""
        return session_id in self._cancelled_sessions

    async def sync(
        self,
        root_path: Path,
        user_id: str,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        conflict_resolution: ConflictResolution = ConflictResolution.ASK,
        extensions: Optional[List[str]] = None,
        post_sync_cleanup: bool = False,
    ) -> Tuple[str, SyncProgress]:
        """
        Main sync method.

        Args:
            root_path: Root directory for sync
            user_id: User ID for database operations
            direction: Sync direction
            conflict_resolution: How to handle conflicts
            extensions: File extensions to sync
            post_sync_cleanup: Whether to unlink files after sync

        Returns:
            Tuple of (session_id, progress)
        """
        start_time = time.time()
        log_counter(
            "sync_engine_sync_attempt",
            labels={
                "direction": direction.value,
                "conflict_resolution": conflict_resolution.value,
                "post_sync_cleanup": str(post_sync_cleanup),
            },
        )

        lexical_root = Path(root_path)
        pinned_root = PinnedSyncRoot(lexical_root)
        pinned_root.__enter__()
        root_path = pinned_root.canonical_root
        try:
            session_id = self._create_sync_session(
                lexical_root,
                direction,
                conflict_resolution,
                user_id,
            )
        except Exception:
            pinned_root.__exit__(None, None, None)
            raise
        progress = self._active_sessions[session_id]

        try:
            logger.info(
                "Starting sync session (direction={})",
                direction.value,
            )

            # Scan directory and get DB notes
            disk_files = self._scan_directory(
                root_path,
                extensions,
                pinned_root=pinned_root,
                progress=progress,
            )
            db_notes = self._get_synced_notes_for_root(
                root_path,
                user_id,
                lexical_root=lexical_root,
                progress=progress,
            )

            progress.total_files = len(set(disk_files.keys()) | set(db_notes.keys()))

            if direction == SyncDirection.DISK_TO_DB:
                await self._sync_disk_to_db(
                    session_id,
                    root_path,
                    disk_files,
                    db_notes,
                    conflict_resolution,
                    user_id,
                    progress,
                    pinned_root,
                )
            elif direction == SyncDirection.DB_TO_DISK:
                await self._sync_db_to_disk(
                    session_id,
                    root_path,
                    disk_files,
                    db_notes,
                    conflict_resolution,
                    user_id,
                    progress,
                    pinned_root,
                )
            else:  # BIDIRECTIONAL
                await self._sync_bidirectional(
                    session_id,
                    root_path,
                    disk_files,
                    db_notes,
                    conflict_resolution,
                    user_id,
                    progress,
                    pinned_root,
                )

            # Update session status
            status = (
                SyncStatus.CANCELLED
                if self.is_cancelled(session_id)
                else SyncStatus.COMPLETED
            )
            summary = {
                "created_notes": len(progress.created_notes),
                "updated_notes": len(progress.updated_notes),
                "created_files": len(progress.created_files),
                "updated_files": len(progress.updated_files),
                "conflicts": len(progress.conflicts),
                # Split out so a session's record can never imply the run
                # settled a conflict it only wrote down (task-19554).
                "conflicts_applied": len(progress.applied_conflicts),
                "conflicts_unresolved": len(progress.unresolved_conflicts),
                "preserved_files": len(progress.preserved_files),
                "errors": len(progress.errors),
                "skipped": len(progress.skipped_items),
            }

            self._update_sync_session(session_id, progress, status, summary)

            # Cleanup
            if session_id in self._cancelled_sessions:
                self._cancelled_sessions.remove(session_id)

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "sync_engine_sync_duration",
                duration,
                labels={"status": "success", "direction": direction.value},
            )
            log_counter(
                "sync_engine_sync_complete",
                labels={
                    "direction": direction.value,
                    "created_notes": str(summary["created_notes"]),
                    "updated_notes": str(summary["updated_notes"]),
                    "created_files": str(summary["created_files"]),
                    "updated_files": str(summary["updated_files"]),
                    "conflicts": str(summary["conflicts"]),
                    "errors": str(summary["errors"]),
                },
            )
            log_histogram("sync_engine_sync_conflicts", len(progress.conflicts))
            log_histogram("sync_engine_sync_errors", len(progress.errors))

            logger.info(f"Sync session {session_id} completed: {summary}")

        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "sync_engine_sync_duration",
                duration,
                labels={"status": "error", "direction": direction.value},
            )
            log_counter(
                "sync_engine_sync_error",
                labels={"direction": direction.value, "error_type": type(e).__name__},
            )

            logger.opt(exception=True).error(f"Sync session {session_id} failed: {e}")
            self._update_sync_session(session_id, progress, SyncStatus.FAILED)
            raise
        finally:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            pinned_root.__exit__(None, None, None)

        return session_id, progress

    async def _sync_disk_to_db(
        self,
        session_id: str,
        root_path: Path,
        disk_files: Dict[Path, SyncFileInfo],
        db_notes: Dict[Path, Dict[str, Any]],
        conflict_resolution: ConflictResolution,
        user_id: str,
        progress: SyncProgress,
        pinned_root: PinnedSyncRoot,
    ):
        """Sync from disk to database.

        task-19554: this direction used to treat "the file changed" as
        sufficient reason to overwrite the note, without ever asking whether
        the NOTE had changed too. A note edited in the app and a file edited
        outside it is the same ``both_changed`` conflict the other two
        directions detect, and the note body went silently. It is now detected,
        preserved, and resolved by the selected strategy like everywhere else.
        """
        for rel_path, file_info in disk_files.items():
            if self.is_cancelled(session_id):
                break

            db_note = db_notes.get(rel_path)

            try:
                if not db_note:
                    # New file on disk -> Create note in DB
                    title = file_info.absolute_path.stem
                    note_id = self.notes_service.add_note(
                        user_id=user_id, title=title, content=file_info.content
                    )

                    if note_id:
                        self._update_note_sync_metadata(
                            note_id, file_info, root_path, user_id, 1
                        )
                        progress.created_notes.append(note_id)
                        logger.info(f"Created note from file: {rel_path}")

                elif file_info.content_hash != db_note.get(
                    "last_synced_disk_file_hash"
                ):
                    last_synced_hash = db_note.get("last_synced_disk_file_hash")
                    db_changed = db_note["content_hash"] != last_synced_hash

                    if db_changed:
                        # Both sides moved since the baseline -- a conflict,
                        # not a one-way pull.
                        conflict = SyncConflict(
                            note_id=db_note["id"],
                            file_path=rel_path,
                            conflict_type="both_changed",
                            db_content=db_note["content"],
                            disk_content=file_info.content,
                            db_hash=db_note["content_hash"],
                            disk_hash=file_info.content_hash,
                        )
                        progress.conflicts.append(conflict)
                        self._record_conflict(session_id, conflict)
                        self._resolve_both_changed(
                            conflict=conflict,
                            conflict_resolution=conflict_resolution,
                            progress=progress,
                            pinned_root=pinned_root,
                            relative_path=rel_path,
                            root_path=root_path,
                            user_id=user_id,
                            db_note=db_note,
                            disk_file=file_info,
                            may_write_disk=False,
                            may_write_db=True,
                            include_title=True,
                        )
                    else:
                        # Only the file changed -> Update note in DB
                        success = self.notes_service.update_note(
                            user_id=user_id,
                            note_id=db_note["id"],
                            update_data={
                                "content": file_info.content,
                                "title": file_info.absolute_path.stem,
                            },
                            expected_version=db_note["version"],
                        )

                        if success:
                            # Get updated version
                            updated_note = self.notes_service.get_note_by_id(
                                user_id, db_note["id"]
                            )
                            if updated_note:
                                self._update_note_sync_metadata(
                                    db_note["id"],
                                    file_info,
                                    root_path,
                                    user_id,
                                    updated_note["version"],
                                )
                                progress.updated_notes.append(db_note["id"])
                                logger.info(f"Updated note from file: {rel_path}")

            except SyncPathError as exc:
                self._record_path_skip(progress, rel_path, exc)
            except Exception as e:
                logger.error(f"Error syncing {rel_path}: {e}")
                progress.errors.append((str(rel_path), e))

            progress.processed_files += 1
            if self.progress_callback:
                self.progress_callback(progress)

        # Check for notes that no longer have files
        for rel_path, db_note in db_notes.items():
            if (
                rel_path not in disk_files
                and not self._path_was_rejected(progress, rel_path)
            ):
                conflict = SyncConflict(
                    note_id=db_note["id"],
                    file_path=rel_path,
                    conflict_type="deleted_on_disk",
                    db_content=db_note["content"],
                    db_hash=db_note.get("content_hash"),
                )
                progress.conflicts.append(conflict)
                self._record_conflict(session_id, conflict)

                # Auto-unlink if not asking. Nothing is destroyed here -- the
                # note keeps its content and simply stops being mirrored --
                # so there is no losing side to preserve, but the outcome is
                # still stamped so the run cannot over-claim.
                if conflict_resolution != ConflictResolution.ASK:
                    try:
                        self._unlink_note_from_sync(db_note["id"], db_note["version"])
                        conflict.applied = True
                        conflict.resolution = RESOLUTION_USE_DISK
                        self._update_conflict_record(
                            conflict, resolution=RESOLUTION_USE_DISK
                        )
                        logger.info(
                            f"Unlinked note {db_note['id']} - file deleted on disk"
                        )
                    except Exception as e:
                        logger.error(f"Error unlinking note {db_note['id']}: {e}")
                        progress.errors.append((str(rel_path), e))

    async def _sync_db_to_disk(
        self,
        session_id: str,
        root_path: Path,
        disk_files: Dict[Path, SyncFileInfo],
        db_notes: Dict[Path, Dict[str, Any]],
        conflict_resolution: ConflictResolution,
        user_id: str,
        progress: SyncProgress,
        pinned_root: PinnedSyncRoot,
    ):
        """Sync from database to disk."""
        for rel_path, db_note in db_notes.items():
            if self.is_cancelled(session_id):
                break
            if self._path_was_rejected(progress, rel_path):
                progress.processed_files += 1
                if self.progress_callback:
                    self.progress_callback(progress)
                continue

            db_content_hash = db_note["content_hash"]

            try:
                if rel_path not in disk_files:
                    # Note in DB but no file -> Create file
                    new_file_info = self._write_file_info(
                        pinned_root,
                        rel_path,
                        db_note["content"],
                    )
                    self._update_note_sync_metadata(
                        db_note["id"],
                        new_file_info,
                        root_path,
                        user_id,
                        db_note["version"],
                    )
                    progress.created_files.append(new_file_info.absolute_path)
                    logger.info("Created sync file from note")

                elif db_content_hash != db_note.get("last_synced_disk_file_hash"):
                    # Note changed in DB -> Update file
                    disk_file = disk_files[rel_path]

                    if db_content_hash != disk_file.content_hash:
                        # Check for conflict
                        if disk_file.content_hash != db_note.get(
                            "last_synced_disk_file_hash"
                        ):
                            # Both changed - conflict!
                            conflict = SyncConflict(
                                note_id=db_note["id"],
                                file_path=rel_path,
                                conflict_type="both_changed",
                                db_content=db_note["content"],
                                disk_content=disk_file.content,
                                db_hash=db_content_hash,
                                disk_hash=disk_file.content_hash,
                            )
                            progress.conflicts.append(conflict)
                            self._record_conflict(session_id, conflict)

                            # Resolve based on strategy. Only DB_WINS had a
                            # branch here before task-19554, so NEWER_WINS
                            # silently declined to push even when the note WAS
                            # the newer side -- and the DB_WINS push destroyed
                            # the file's text with no copy kept.
                            self._resolve_both_changed(
                                conflict=conflict,
                                conflict_resolution=conflict_resolution,
                                progress=progress,
                                pinned_root=pinned_root,
                                relative_path=rel_path,
                                root_path=root_path,
                                user_id=user_id,
                                db_note=db_note,
                                disk_file=disk_file,
                                may_write_disk=True,
                                may_write_db=False,
                            )
                        else:
                            # Only DB changed
                            new_file_info = self._write_file_info(
                                pinned_root,
                                rel_path,
                                db_note["content"],
                            )
                            progress.updated_files.append(
                                new_file_info.absolute_path
                            )
                            self._update_note_sync_metadata(
                                db_note["id"],
                                new_file_info,
                                root_path,
                                user_id,
                                db_note["version"],
                            )
                            logger.info("Updated sync file from note")

            except SyncPathError as exc:
                self._record_path_skip(progress, rel_path, exc)
            except Exception as exc:
                logger.error(
                    "Error syncing note to disk (error_type={})",
                    type(exc).__name__,
                )
                progress.errors.append((f"Note {db_note['id']}", exc))

            progress.processed_files += 1
            if self.progress_callback:
                self.progress_callback(progress)

    async def _sync_bidirectional(
        self,
        session_id: str,
        root_path: Path,
        disk_files: Dict[Path, SyncFileInfo],
        db_notes: Dict[Path, Dict[str, Any]],
        conflict_resolution: ConflictResolution,
        user_id: str,
        progress: SyncProgress,
        pinned_root: PinnedSyncRoot,
    ):
        """Bidirectional sync between disk and database."""
        all_paths = set(disk_files.keys()) | set(db_notes.keys())

        for rel_path in all_paths:
            if self.is_cancelled(session_id):
                break
            if self._path_was_rejected(progress, rel_path):
                progress.processed_files += 1
                if self.progress_callback:
                    self.progress_callback(progress)
                continue

            disk_file = disk_files.get(rel_path)
            db_note = db_notes.get(rel_path)

            try:
                if disk_file and not db_note:
                    # Only on disk -> Create in DB
                    title = disk_file.absolute_path.stem
                    note_id = self.notes_service.add_note(
                        user_id=user_id, title=title, content=disk_file.content
                    )

                    if note_id:
                        self._update_note_sync_metadata(
                            note_id, disk_file, root_path, user_id, 1
                        )
                        progress.created_notes.append(note_id)
                        logger.info(f"Created note from file: {rel_path}")

                elif not disk_file and db_note:
                    # Only in DB -> Create on disk or handle deletion
                    conflict = SyncConflict(
                        note_id=db_note["id"],
                        file_path=rel_path,
                        conflict_type="deleted_on_disk",
                        db_content=db_note["content"],
                    )
                    progress.conflicts.append(conflict)
                    self._record_conflict(session_id, conflict)

                    # Auto-resolve based on strategy. Only DB_WINS acts; the
                    # other strategies genuinely have no defined answer for a
                    # vanished file (there is no disk mtime to compare, and
                    # "the disk wins" would mean guessing that the user meant
                    # to delete the note). They leave it OPEN -- see
                    # ``applied`` -- rather than reporting it as settled.
                    if conflict_resolution == ConflictResolution.DB_WINS:
                        # Recreate file
                        new_file_info = self._write_file_info(
                            pinned_root,
                            rel_path,
                            db_note["content"],
                        )
                        self._update_note_sync_metadata(
                            db_note["id"],
                            new_file_info,
                            root_path,
                            user_id,
                            db_note["version"],
                        )
                        progress.created_files.append(new_file_info.absolute_path)
                        conflict.applied = True
                        conflict.resolution = RESOLUTION_USE_DB
                        self._update_conflict_record(
                            conflict, resolution=RESOLUTION_USE_DB
                        )

                elif disk_file and db_note:
                    # Exists in both - check for changes
                    db_content_hash = db_note["content_hash"]
                    disk_hash = disk_file.content_hash
                    last_synced_hash = db_note.get("last_synced_disk_file_hash")

                    db_changed = db_content_hash != last_synced_hash
                    disk_changed = disk_hash != last_synced_hash

                    if db_changed and not disk_changed:
                        # Only DB changed -> Update disk
                        new_file_info = self._write_file_info(
                            pinned_root,
                            rel_path,
                            db_note["content"],
                        )
                        self._update_note_sync_metadata(
                            db_note["id"],
                            new_file_info,
                            root_path,
                            user_id,
                            db_note["version"],
                        )
                        progress.updated_files.append(new_file_info.absolute_path)

                    elif not db_changed and disk_changed:
                        # Only disk changed -> Update DB
                        success = self.notes_service.update_note(
                            user_id=user_id,
                            note_id=db_note["id"],
                            update_data={
                                "content": disk_file.content,
                                "title": disk_file.absolute_path.stem,
                            },
                            expected_version=db_note["version"],
                        )

                        if success:
                            updated_note = self.notes_service.get_note_by_id(
                                user_id, db_note["id"]
                            )
                            if updated_note:
                                self._update_note_sync_metadata(
                                    db_note["id"],
                                    disk_file,
                                    root_path,
                                    user_id,
                                    updated_note["version"],
                                )
                            progress.updated_notes.append(db_note["id"])

                    elif db_changed and disk_changed:
                        # Both changed - CONFLICT!
                        conflict = SyncConflict(
                            note_id=db_note["id"],
                            file_path=rel_path,
                            conflict_type="both_changed",
                            db_content=db_note["content"],
                            disk_content=disk_file.content,
                            db_hash=db_content_hash,
                            disk_hash=disk_hash,
                        )
                        progress.conflicts.append(conflict)
                        self._record_conflict(session_id, conflict)

                        # Auto-resolve if not asking. Before task-19554 only
                        # NEWER_WINS was implemented here, and it overwrote the
                        # losing side with no copy kept; DB_WINS and DISK_WINS
                        # recorded the conflict and applied nothing at all.
                        self._resolve_both_changed(
                            conflict=conflict,
                            conflict_resolution=conflict_resolution,
                            progress=progress,
                            pinned_root=pinned_root,
                            relative_path=rel_path,
                            root_path=root_path,
                            user_id=user_id,
                            db_note=db_note,
                            disk_file=disk_file,
                            may_write_disk=True,
                            may_write_db=True,
                        )

            except SyncPathError as exc:
                self._record_path_skip(progress, rel_path, exc)
            except Exception as exc:
                logger.error(
                    "Error syncing entry (error_type={})",
                    type(exc).__name__,
                )
                progress.errors.append((str(rel_path), exc))

            progress.processed_files += 1
            if self.progress_callback:
                self.progress_callback(progress)


#
# End of sync_engine.py
########################################################################################################################
