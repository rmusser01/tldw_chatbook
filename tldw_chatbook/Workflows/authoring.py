"""Lazy application-owned authoring and explicit, bounded JSON exchange."""

import asyncio
import stat
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Utils.input_validation import validate_workflow_name
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
        self._quit_preparing = False
        self._quit_task: asyncio.Task | None = None
        self._quit_tasks: set[asyncio.Task] = set()

    async def open(self) -> None:
        """Initialize once; a cancelled waiter cannot abandon construction."""
        if self._closed or self._closing or self._quit_preparing:
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
        if self._closed or self._closing or self._quit_preparing:
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

    async def prepare_quit(self) -> None:
        """Retain fallible persistence behind reversible admission fences.

        Keep the same document, draft and database owners available for abort.
        A cancelled waiter cannot abandon the accepted preparation operation.
        """
        if self._closed or self._closing:
            raise DraftWriteFailed("Workflow authoring is closing or closed")
        self._quit_preparing = True
        # Abort may have released the child fence and admitted more work. Each
        # attempt must flush that work and establish its own child barrier.
        task = asyncio.create_task(self._prepare_quit())
        self._quit_task = task
        self._quit_tasks.add(task)

        def settled(task: asyncio.Task) -> None:
            self._quit_tasks.discard(task)
            if not task.cancelled():
                task.exception()

        task.add_done_callback(settled)
        await asyncio.shield(task)

    async def _prepare_quit(self) -> None:
        await self.flush()
        # An aborted or superseded flush must not fence the newer attempt's
        # accepted work before that attempt has finished draining it.
        if (
            self._quit_preparing
            and self._quit_task is asyncio.current_task()
            and self.drafts
        ):
            await self.drafts.prepare_quit()

    def abort_quit(self) -> None:
        """Reopen this graph after an uncommitted quit, without replacing it."""
        self._quit_preparing = False
        if self.drafts:
            self.drafts.abort_quit()

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
        if self._quit_tasks:
            await asyncio.shield(
                asyncio.gather(*self._quit_tasks, return_exceptions=True)
            )
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
        """Create and select a named definition, retaining the accepted write.

        Args:
            name: New-workflow text, at most 256 raw Python characters. Surrounding
                whitespace is removed; the result must not be empty.

        Returns:
            The saved revision, selected as the current draft base.

        Raises:
            ValueError: The name is not text, is too long or becomes empty.
            DraftWriteFailed: Authoring is closing/closed or the current draft
                could not be durably flushed before creation.
            RevisionConflict: A generated portable identity already exists.
            OSError: Private storage could not be opened or accessed.
            sqlite3.Error: Storage setup, revision persistence or selection fails.

        Cancellation of a waiter does not abandon an accepted creation. A
        selection failure after insertion does not roll back that saved revision.
        """
        name = validate_workflow_name(name)
        await self.open()
        raw = self.documents.edit_field('{"steps":[],"inputs":{}}', "/name", name)
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
        """Import an explicitly selected private JSON file; never execute it.

        Args:
            path: Regular, nonlinked .json file separate from the workflow
                database and sidecars, subject to the stable-file contract.

        Returns:
            The imported saved revision, selected as the current draft base.
            Imported names are not subject to the New workflow dialog's limit.

        Raises:
            ValueError: The path is refused, the file exceeds 16 MiB, or UTF-8
                decoding fails.
            InvalidDraft: The definition fails local structural admission.
            RevisionConflict: Its portable identity conflicts with saved content.
            DraftWriteFailed: Authoring is closing/closed or the current draft
                cannot be durably flushed.
            OSError: Private storage or the selected file cannot be accessed.
            sqlite3.Error: Storage setup, import persistence or selection fails.

        Accepted work survives waiter cancellation. A saved revision can remain
        if selection subsequently fails; this method never imports run authority.
        """
        await self.open()
        return await self._retain(self._import(path))

    async def _import(self, path: Path) -> Revision:
        raw = await asyncio.to_thread(self._read_import, path)
        return await self._create(raw)

    async def export_file(self, path: Path, revision: Revision) -> None:
        """Export only an exact saved definition, excluding local draft state.

        Args:
            path: Separate .json destination, atomically replaced with a private
                file under the approved stable-file exchange contract.
            revision: Exact saved revision, including its workflow/revision IDs
                and raw bytes; local edits or fabricated revisions are refused.

        Raises:
            ValueError: The destination is refused or the supplied revision
                differs from the saved revision.
            RevisionConflict: The requested saved identity does not exist.
            DraftWriteFailed: Authoring is closing or closed.
            OSError: Private storage or the destination cannot be accessed/written.
            sqlite3.Error: Storage setup or the saved-revision read fails.

        Accepted export survives waiter cancellation; close waits for settlement.
        """
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
