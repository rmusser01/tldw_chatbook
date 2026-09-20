"""App-owned recoverable buffer; screen removal cannot cancel durable writes."""

import asyncio
from collections.abc import Callable

from tldw_chatbook.Workflows.document_service import FRAGMENT_ERROR, DocumentService
from tldw_chatbook.Workflows.models import (
    Draft,
    DraftConflict,
    DraftWriteFailed,
    FieldEdit,
    InvalidDraft,
    Revision,
    RevisionConflict,
)

DRAFT_DEBOUNCE_SECONDS = 0.5


class DraftSession:
    """Own one selected draft and retain its writes across fresh screen lifetimes.

    Construct/use on the app loop. The app must await close before closing the
    injected document store. Observer callbacks receive only the current state;
    they must not mutate the owner or perform I/O.
    """

    def __init__(self, documents: DocumentService) -> None:
        self.documents = documents
        self.current: Draft | None = None
        self.base: Revision | None = None
        self.status = "No workflow selected"
        self._durable: Draft | None = None
        self._persisted = False
        self._generation = 0
        self._timer: asyncio.TimerHandle | None = None
        self._flush_task: asyncio.Task | None = None
        self._save_task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None
        self._editing_locked = False
        self._confirmation_version = 0
        self._selections: set[asyncio.Task] = set()
        self._transition = asyncio.Lock()
        self._listeners: list[Callable[[], None]] = []
        self._closed = False
        self._closing = False
        self._quit_preparing = False
        self._field_edit: FieldEdit | None = None
        self._field_checkpoints: list[tuple[Draft, FieldEdit]] = []
        self._durable_field_edits: dict[tuple[str, str], tuple[Draft, FieldEdit]] = {}

    @property
    def field_edit(self) -> FieldEdit | None:
        """Only expose provenance that still reproduces the exact current buffer."""
        edit = self._field_edit
        if edit and self.current and self.base:
            try:
                if (
                    self.documents.edit_fragment(
                        self.base, edit, self.current.generation
                    )
                    == self.current
                ):
                    return edit
            except (DraftConflict, InvalidDraft):
                pass
        return None

    @property
    def raw_repair_required(self) -> bool:
        """Protected field bytes require explicit whole-document adoption."""
        return bool(self.current and self.current.error == FRAGMENT_ERROR)

    def _checkpoint_field(self) -> None:
        # Retain an ownership transition, not every keystroke. Debounce still
        # coalesces typing within one field. Later raw edits must not overtake
        # the proof that protected (or repaired) their durable predecessor.
        if self._field_edit and self.current != self._durable:
            self._field_checkpoints.append((self.current, self._field_edit))
        self._field_edit = None

    @property
    def editing_locked(self) -> bool:
        """True until the accepted physical recovery settles, even if cancelled."""
        return self._editing_locked

    @property
    def confirmation_version(self) -> int:
        """Monotonic edit/selection intent counter; never persisted as document data."""
        return self._confirmation_version

    def _check_editable(self) -> None:
        if self._closed or self._closing or self._quit_preparing or self.editing_locked:
            raise DraftWriteFailed(
                "Draft owner is closed or recovery is still in progress"
            )

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Observe state; returned release is owned by the subscribing screen."""
        self._listeners.append(callback)

        def release():
            if callback in self._listeners:
                self._listeners.remove(callback)

        return release

    def _publish(self, status: str) -> None:
        self.status = status
        for listener in tuple(self._listeners):
            listener()

    async def select(self, workflow_id: str, base_revision_id: str) -> Draft:
        """Flush before changing exact identities, including selecting this base."""
        self._check_editable()
        self._confirmation_version += 1
        task = asyncio.create_task(self._select(workflow_id, base_revision_id))
        self._selections.add(task)

        def settled(task):
            self._selections.discard(task)
            if not task.cancelled():
                task.exception()

        task.add_done_callback(settled)
        return await asyncio.shield(task)

    async def _select(self, workflow_id: str, base_revision_id: str) -> Draft:
        async with self._transition:
            if self._closed:
                raise DraftWriteFailed("Draft owner is closed")
            if self.current:
                await self.flush()
                if (workflow_id, base_revision_id) == (
                    self.current.workflow_id,
                    self.current.base_revision_id,
                ):
                    return self.current
            base = await asyncio.to_thread(
                self.documents.get_revision, workflow_id, base_revision_id
            )
            recovered = await asyncio.to_thread(
                self.documents.get_draft, workflow_id, base_revision_id
            )
            # A user may still type while the target read is pending.
            if self.current:
                await self.flush()
            self.base = base
            self._field_edit = None
            self.current = recovered or Draft(
                workflow_id, base_revision_id, 0, base.raw_json, base.raw_json, None
            )
            self._durable = self.current
            known = self._durable_field_edits.get((workflow_id, base_revision_id))
            self._field_edit = known[1] if known and known[0] == self.current else None
            self._persisted = recovered is not None
            self._generation = self.current.generation
            self._publish("Draft recovered" if recovered else "Based on saved revision")
            return self.current

    def update(self, raw_text: str) -> Draft:
        """Validate raw editing synchronously and schedule the shared debounce.

        Args:
            raw_text: Complete Advanced JSON buffer. Invalid JSON or document
                structure remains recoverable with its last valid projection.

        Returns:
            The current draft for unchanged text, otherwise the new draft with
            validation feedback. A scheduled write is not a durability guarantee.

        Raises:
            InvalidDraft: The raw buffer violates text admission limits.
            DraftConflict: The prior draft belongs to a different base or the next
                generation exceeds the SQLite integer range.
            DraftWriteFailed: No editable selection exists or the owner is locked,
                closing, or closed.
        """
        self._check_editable()
        if self.base is None or self.current is None:
            raise DraftWriteFailed("No editable draft owner is available")
        if raw_text == self.current.raw_text:
            return self.current
        candidate = self.documents.validate_draft(
            self.base, raw_text, self._generation + 1, self.current
        )
        self._checkpoint_field()
        return self._updated(candidate)

    def update_field(self, pointer: str, text: str) -> Draft:
        """Keep an incomplete JSON field repairable without enabling stale fields."""
        self._check_editable()
        if self.current is None or self.base is None:
            raise DraftWriteFailed("No editable draft owner is available")
        proof = self.field_edit
        if self.current.error and (proof is None or pointer != proof.pointer):
            raise InvalidDraft(
                "Repair the active field or explicitly repair Advanced JSON"
            )
        if proof and pointer == proof.pointer:
            if text == proof.text:
                return self.current
            edit = FieldEdit(proof.source, pointer, text)
        else:
            self._checkpoint_field()
            edit = FieldEdit(self.current, pointer, text)
        candidate = self.documents.edit_fragment(self.base, edit, self._generation + 1)
        self._field_edit = edit
        return self._updated(candidate)

    def _updated(self, candidate: Draft) -> Draft:
        self._generation = candidate.generation
        self.current = candidate
        self._confirmation_version += 1
        self._cancel_timer()
        self._timer = asyncio.get_running_loop().call_later(
            DRAFT_DEBOUNCE_SECONDS, self._debounced_flush
        )
        self._publish(f"Pending — saved after {DRAFT_DEBOUNCE_SECONDS * 1000:g} ms")
        return candidate

    def _cancel_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _debounced_flush(self) -> None:
        self._timer = None
        self._ensure_flush()

    def _ensure_flush(self) -> asyncio.Task:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._write_until_current())
            # Keep the exception observable via flush/status without an unhandled
            # task exception when debounce was the only caller.
            self._flush_task.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
        return self._flush_task

    async def _write_until_current(self) -> Draft:
        if self.current is None:
            raise DraftWriteFailed("No draft selected")
        while (
            self._field_checkpoints
            or not self._persisted
            or self.current != self._durable
        ):
            checkpoint = bool(self._field_checkpoints)
            candidate, proof = (
                self._field_checkpoints[0]
                if checkpoint
                else (self.current, self._field_edit)
            )
            self._publish("Saving locally…")
            try:
                durable = await asyncio.to_thread(
                    self.documents.put_draft,
                    candidate.workflow_id,
                    candidate.base_revision_id,
                    candidate.raw_text,
                    candidate.generation,
                    last_valid_json=candidate.last_valid_json,
                    field_edit=proof,
                )
            except Exception as error:
                self._publish("Not saved locally — Retry")
                raise DraftWriteFailed(
                    "Draft persistence failed; pending changes remain in memory"
                ) from error
            if checkpoint:
                self._field_checkpoints.pop(0)
            # Only this owner changes selection, and selection joins this task.
            if (candidate.workflow_id, candidate.base_revision_id) == (
                self.current.workflow_id,
                self.current.base_revision_id,
            ):
                self._durable = durable
                key = (durable.workflow_id, durable.base_revision_id)
                if proof and durable.error:
                    self._durable_field_edits[key] = (durable, proof)
                else:
                    self._durable_field_edits.pop(key, None)
                self._persisted = True
                if candidate.generation == self.current.generation:
                    self.current = durable
                    if not durable.error:
                        self._field_edit = None
                    self._publish("Saved locally")
        return self.current

    async def flush(self) -> Draft:
        """Await all current generations; caller cancellation cannot cancel I/O."""
        if self._recovery_task and asyncio.current_task() is not self._recovery_task:
            # A failed copy leaves the original durable buffer intact. Joining it
            # still permits navigation/close to flush that coherent old state.
            await asyncio.shield(
                asyncio.gather(self._recovery_task, return_exceptions=True)
            )
        self._cancel_timer()
        return await asyncio.shield(self._ensure_flush())

    async def save_revision(self) -> Revision:
        """Save the flushed generation; newer typing keeps its existing owner."""
        self._check_editable()
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._save_revision())
            self._save_task.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
        return await asyncio.shield(self._save_task)

    async def _save_revision(self) -> Revision:
        async with self._transition:
            draft = await self.flush()
            if draft.error:
                raise InvalidDraft(draft.error)
            revision = await asyncio.to_thread(
                self.documents.save_revision,
                draft.workflow_id,
                draft.base_revision_id,
                draft.generation,
            )
            if self.current == draft:
                self.base = revision
                self.current = Draft(
                    revision.workflow_id,
                    revision.revision_id,
                    0,
                    revision.raw_json,
                    revision.raw_json,
                    None,
                )
                self._durable = self.current
                self._persisted = False
                self._generation = 0
                self._field_edit = None
                self._confirmation_version += 1
                self._publish("Revision saved")
            else:
                self._publish("Revision saved; newer edits retained on original base")
            return revision

    async def recover_to_head(
        self, source: Draft, head: Revision, *, confirmation_version: int
    ) -> Draft:
        """Apply one explicit confirmation; retain the operation beyond its caller."""
        self._check_editable()
        if (
            self._transition.locked()
            or self._selections
            or (self._save_task and not self._save_task.done())
        ):
            raise DraftWriteFailed(
                "Wait for the current draft operation, then confirm recovery"
            )
        self._check_confirmation(source, confirmation_version)
        if source.error:
            raise InvalidDraft("Repair raw JSON before copying the draft")
        self._editing_locked = True
        self._recovery_task = asyncio.create_task(
            self._recover_to_head(source, head, confirmation_version)
        )
        self._recovery_task.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )
        self._publish("Copying draft onto saved head — editing temporarily locked")
        return await asyncio.shield(self._recovery_task)

    def _check_confirmation(self, source: Draft | None, version: int) -> None:
        if source != self.current or version != self.confirmation_version:
            raise DraftConflict("Draft or selection changed; open a new confirmation")

    async def _recover_to_head(
        self, source: Draft, head: Revision, version: int
    ) -> Draft:
        status = "Recovery failed; original draft retained. Retry the recovery."
        try:
            async with self._transition:
                self._check_confirmation(source, version)
                await self.flush()
                copied = await asyncio.to_thread(
                    self.documents.copy_draft_to_head, source, head
                )
                self.base = head
                self.current = self._durable = copied
                self._generation = copied.generation
                self._field_edit = None
                self._persisted = True
                self._confirmation_version += 1
                status = "Draft copied onto saved head; Save revision when ready"
                return copied
        except ValueError as error:
            status = str(error)
            raise
        except Exception as error:
            raise DraftWriteFailed(status) from error
        finally:
            self._editing_locked = False
            self._publish(status)

    async def repair_raw(self, source: Draft, *, confirmation_version: int) -> Draft:
        """Explicitly accept the whole raw document, retaining physical ownership."""
        self._check_editable()
        if (
            self._transition.locked()
            or self._selections
            or (self._save_task and not self._save_task.done())
        ):
            raise DraftWriteFailed(
                "Wait for the current draft operation, then confirm repair"
            )
        self._check_confirmation(source, confirmation_version)
        self._editing_locked = True
        self._recovery_task = asyncio.create_task(
            self._repair_raw(source, confirmation_version)
        )
        self._recovery_task.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )
        self._publish("Accepting repaired Advanced JSON — editing temporarily locked")
        return await asyncio.shield(self._recovery_task)

    async def copy_revision_to_head(self, source: Revision, head: Revision) -> Draft:
        """Keep historical copying locked/owned through physical settlement."""
        self._check_editable()
        if (
            self._transition.locked()
            or self._selections
            or (self._save_task and not self._save_task.done())
        ):
            raise DraftWriteFailed(
                "Wait for the current draft operation before editing history"
            )
        self._editing_locked = True
        self._recovery_task = asyncio.create_task(self._copy_revision(source, head))
        self._recovery_task.add_done_callback(
            lambda task: task.exception() if not task.cancelled() else None
        )
        self._publish("Copying historical revision — editing temporarily locked")
        return await asyncio.shield(self._recovery_task)

    async def _copy_revision(self, source: Revision, head: Revision) -> Draft:
        status = "Historical copy failed; original draft retained."
        try:
            async with self._transition:
                await self.flush()
                copied = await asyncio.to_thread(
                    self.documents.copy_revision_to_head, source, head
                )
                self.base = head
                self.current = self._durable = copied
                self._generation = copied.generation
                self._persisted = True
                self._field_edit = None
                self._confirmation_version += 1
                status = "Historical content copied; Save revision when ready"
                return copied
        except ValueError as error:
            status = str(error)
            raise
        except Exception as error:
            raise DraftWriteFailed(status) from error
        finally:
            self._editing_locked = False
            self._publish(status)

    async def _repair_raw(self, source: Draft, version: int) -> Draft:
        status = "Raw repair failed; protected draft retained. Retry repair."
        try:
            async with self._transition:
                self._check_confirmation(source, version)
                await self.flush()
                repaired = self.documents.repair_draft(
                    self.base, source, self._generation + 1
                )
                durable = await asyncio.to_thread(
                    self.documents.put_draft,
                    repaired.workflow_id,
                    repaired.base_revision_id,
                    repaired.raw_text,
                    repaired.generation,
                    repair_source=source,
                )
                self.current = self._durable = durable
                self._generation = durable.generation
                self._persisted = True
                self._field_edit = None
                self._confirmation_version += 1
                status = "Repaired Advanced JSON accepted; Save revision when ready"
                return durable
        except ValueError as error:
            status = str(error)
            raise
        except Exception as error:
            raise DraftWriteFailed(status) from error
        finally:
            self._editing_locked = False
            self._publish(status)

    async def discard_pending(self) -> Draft:
        """Restore durable text after the caller confirms losing pending edits.

        Returns:
            The last durable draft for the selected base, including retained
            invalid text, after an existing flush settles. The owner also restores
            any associated field provenance; it is not part of the returned Draft.
            A failed flush does not prevent restoring that durable snapshot.

        Raises:
            DraftConflict: Edits or selection supersede the discard, or the durable
                snapshot belongs to another base. Newer state is left intact.
            DraftWriteFailed: No durable selection exists or the owner is locked,
                closing, or closed.
        """
        self._check_editable()
        self._confirmation_version += 1
        source, version = self.current, self.confirmation_version
        async with self._transition:
            self._check_editable()
            self._check_confirmation(source, version)
            self._cancel_timer()
            if self._flush_task and not self._flush_task.done():
                try:
                    await asyncio.shield(self._flush_task)
                except DraftWriteFailed:
                    pass
                self._check_confirmation(source, version)
            if self._durable is None or self.base is None:
                raise DraftWriteFailed("No durable buffer available")
            if (self._durable.workflow_id, self._durable.base_revision_id) != (
                self.base.workflow_id,
                self.base.revision_id,
            ):
                raise DraftConflict("Durable draft belongs to another base")
            self.current = self._durable
            self._field_checkpoints.clear()
            known = self._durable_field_edits.get(
                (self.current.workflow_id, self.current.base_revision_id)
            )
            self._field_edit = known[1] if known and known[0] == self.current else None
            self._publish("Pending changes discarded; durable draft restored")
            return self.current

    async def discard_draft(self) -> Draft:
        """Durably return to base after the caller confirms discarding the draft.

        Returns:
            The persisted base-content draft, with protected raw repair completed
            when needed. No new immutable workflow revision is created.

        Raises:
            InvalidDraft: Base content cannot pass text admission or raw repair.
            DraftConflict: Edits or selection supersede the discard or retained
                repair. Newer edits are not accepted on the old confirmation.
            DraftWriteFailed: Selection is unavailable, the owner is locked or
                closed, or persistence fails. Pending state remains recoverable.
        """
        self._check_editable()
        self._confirmation_version += 1
        source, version = self.current, self.confirmation_version
        async with self._transition:
            self._check_editable()
            self._check_confirmation(source, version)
            if self.base is None:
                raise DraftWriteFailed("No draft selected")
            source = self.update(self.base.raw_json)
            version = self.confirmation_version
            draft = await self.flush()
            self._check_confirmation(source, version)
            if not self.raw_repair_required:
                return draft
        # repair_raw acquires this same non-reentrant lock. There is no await
        # between releasing it and admitting the existing retained repair task.
        return await self.repair_raw(draft, confirmation_version=version)

    async def prepare_quit(self) -> None:
        """Fence edits and settle persistence without closing this draft owner.

        The authoring owner retains this operation if its quit waiter cancels.
        Abort explicitly to reopen edits; no buffer or provenance is replaced.
        """
        if self._closed or self._closing:
            raise DraftWriteFailed("Draft owner is closing or closed")
        self._quit_preparing = True
        await self._drain_accepted()

    def abort_quit(self) -> None:
        """Release only the reversible quit fence, never a permanent close."""
        self._quit_preparing = False

    async def _drain_accepted(self) -> None:
        if self._recovery_task:
            await asyncio.shield(
                asyncio.gather(self._recovery_task, return_exceptions=True)
            )
        if self._selections:
            await asyncio.shield(
                asyncio.gather(*self._selections, return_exceptions=True)
            )
        if self._save_task and not self._save_task.done():
            try:
                await asyncio.shield(self._save_task)
            except (InvalidDraft, RevisionConflict, DraftConflict):
                # A rejected revision is not lost draft text. The final
                # flush must still run and its write failures must escape.
                pass
        if self.current:
            await self.flush()

    async def close(self) -> None:
        """Drain authoring before app-owned store teardown; failure is retryable."""
        self._closing = True
        try:
            await self._drain_accepted()
            self._closed = True
        finally:
            self._closing = False
