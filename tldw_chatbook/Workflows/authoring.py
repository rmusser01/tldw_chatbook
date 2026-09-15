"""Lazy application-owned authoring and explicit, bounded JSON exchange."""

import asyncio
import stat
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.private_paths import (
    atomic_private_write_text,
    lexical_path,
    open_private_binary,
)
from tldw_chatbook.Workflows.document_service import MAX_DOCUMENT_BYTES, DocumentService
from tldw_chatbook.Workflows.draft_session import DraftSession
from tldw_chatbook.Workflows.models import DraftWriteFailed, Revision

_T = TypeVar("_T")


class WorkflowAuthoring:
    """Retain accepted setup/exchange until drafts and storage can close.

    Construct on the app loop. The path factory runs only on first open, off
    the UI thread. Failed persistence keeps the same interactive draft owner.
    """

    def __init__(self, path_factory: Callable[[], Path]) -> None:
        self._path_factory = path_factory
        self._path: Path | None = None
        self._db: WorkflowsDB | None = None
        self.documents: DocumentService | None = None
        self.drafts: DraftSession | None = None
        self._opening: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._pending: set[asyncio.Task] = set()
        self._closed = False

    async def open(self) -> None:
        """Initialize once; a cancelled waiter cannot abandon construction."""
        if self._closed or self._closing:
            raise DraftWriteFailed("Workflow authoring is closing or closed")
        if self._opening is None or (
            self._opening.done() and self._opening.exception() is not None
        ):
            self._opening = asyncio.create_task(self._open())
        await asyncio.shield(self._opening)

    async def _open(self) -> None:
        self._path = await asyncio.to_thread(self._path_factory)
        self._db = await asyncio.to_thread(WorkflowsDB, self._path)
        self.documents = DocumentService(self._db)
        self.drafts = DraftSession(self.documents)

    async def _retain(self, operation: Coroutine[Any, Any, _T]) -> _T:
        if self._closed or self._closing:
            operation.close()
            raise DraftWriteFailed("Workflow authoring is closing or closed")
        task = asyncio.create_task(operation)
        self._pending.add(task)

        def settled(task: asyncio.Task) -> None:
            self._pending.discard(task)
            if not task.cancelled():
                task.exception()

        task.add_done_callback(settled)
        return await asyncio.shield(task)

    async def flush(self) -> None:
        """Drain accepted work without initializing an unused store."""
        if self._opening:
            try:
                await asyncio.shield(self._opening)
            except Exception:
                if self._db is not None:
                    raise
                # Setup refused before a draft owner existed. There are no
                # pending edits to protect, so this must not trap app quit.
                return
        if self._pending:
            await asyncio.shield(asyncio.gather(*self._pending))
        if self.drafts and self.drafts.current:
            await self.drafts.flush()

    async def close(self) -> None:
        """Drain before DB close; a failed flush leaves the buffer retryable."""
        if self._closed:
            return
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())

            def settled(task: asyncio.Task) -> None:
                if not task.cancelled():
                    task.exception()
                if self._closing is task and not self._closed:
                    self._closing = None

            # Settlement belongs to the retained operation, not a waiter that
            # may be cancelled or race another waiter after the same failure.
            self._closing.add_done_callback(settled)
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        if self._opening:
            try:
                await asyncio.shield(self._opening)
            except Exception:
                if self._db is not None:
                    raise
                self._closed = True
                return
        if self._pending:
            await asyncio.shield(asyncio.gather(*self._pending, return_exceptions=True))
        if self.drafts:
            await self.drafts.close()
        if self._db:
            await asyncio.to_thread(self._db.close)
        self._closed = True

    async def create(self, name: str) -> Revision:
        """Create and select a named definition, retaining the accepted write."""
        if not name.strip():
            raise ValueError("Name the workflow before creating it")
        await self.open()
        raw = self.documents.edit_field(
            '{"steps":[],"inputs":{}}', "/name", name.strip()
        )
        return await self._retain(self._create(raw))

    async def _create(self, raw: str) -> Revision:
        if self.drafts.current:
            await self.drafts.flush()
        revision = await asyncio.to_thread(self.documents.create, raw)
        await self.drafts.select(revision.workflow_id, revision.revision_id)
        return revision

    def _exchange_path(self, path: Path) -> Path:
        """Reject visible database aliases without opening a protected inode.

        This metadata preflight is not a lease against concurrent path changes.
        Generic private-file checks still own normal file access and hardening.
        """
        selected = lexical_path(validate_path_simple(path, probe_existing=False))
        if selected.suffix.lower() != ".json" or selected == lexical_path(self._path):
            raise ValueError("Select a separate .json definition file")
        try:
            entry = selected.lstat()
        except FileNotFoundError:
            return selected
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise ValueError("Select a regular .json definition file without links")
        # Even a refused generic read can close the inode and release POSIX
        # SQLite locks. Only stat here, including for aliases via parent paths.
        for suffix in ("", "-journal", "-wal", "-shm"):
            try:
                protected = Path(str(self._path) + suffix).stat()
            except FileNotFoundError:
                continue
            if (entry.st_dev, entry.st_ino) == (protected.st_dev, protected.st_ino):
                raise ValueError(
                    "Select a .json definition separate from database files"
                )
        return selected

    def _read_import(self, path: Path) -> str:
        with open_private_binary(self._exchange_path(path)) as opened:
            content = opened.stream.read(MAX_DOCUMENT_BYTES + 1)
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Workflow file exceeds the 16 MiB size limit")
        return content.decode("utf-8")

    async def import_file(self, path: Path) -> Revision:
        """Import an explicitly selected private JSON file; never execute it."""
        await self.open()
        return await self._retain(self._import(path))

    async def _import(self, path: Path) -> Revision:
        raw = await asyncio.to_thread(self._read_import, path)
        return await self._create(raw)

    async def export_file(self, path: Path, revision: Revision) -> None:
        """Export only an exact saved definition, excluding local draft state."""
        await self.open()
        await self._retain(self._export(path, revision))

    async def _export(self, path: Path, revision: Revision) -> None:
        saved = await asyncio.to_thread(
            self.documents.get_revision, revision.workflow_id, revision.revision_id
        )
        if saved != revision:
            raise ValueError("Select an exact saved revision before exporting")
        selected = await asyncio.to_thread(self._exchange_path, path)
        await asyncio.to_thread(atomic_private_write_text, selected, saved.raw_json)
