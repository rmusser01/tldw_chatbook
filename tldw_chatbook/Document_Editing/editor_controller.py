"""Revision-safe local checks and native edits; hosts retain all save authority."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import cast

from textual.document._document import Document
from textual.widgets import TextArea

from tldw_chatbook.Widgets.document_editing_bridge import TextAreaFormatBridge

from .models import CheckResult, DocumentRequest, FormatResult
from .scheduler import DocumentCheckScheduler
from .worker import DocumentWorker


class DocumentEditorController:
    """Observe canonical snapshots through one active app-owned worker lease."""

    def __init__(
        self,
        worker: DocumentWorker,
        editor: TextArea,
        *,
        snapshot: Callable[[], DocumentRequest | None],
        editable: Callable[[], bool],
        publish: Callable[
            [DocumentRequest | None, CheckResult | None, str, bool], None
        ],
        map_format: (
            Callable[[FormatResult, DocumentRequest], FormatResult] | None
        ) = None,
    ) -> None:
        self._worker = worker
        self.editor = editor
        self._snapshot = snapshot
        self._editable = editable
        self._publish = publish
        self._map_format = map_format
        self.bridge = TextAreaFormatBridge(editor)
        self.request: DocumentRequest | None = None
        self.result: CheckResult | None = None
        self._lease: DocumentCheckScheduler | None = None
        self._acquiring: asyncio.Task[None] | None = None
        self._generation = 0
        self._request_id = 0
        self._was_editable = False
        self._pending_immediate = False
        self.retired = False

    def _current(self, request: DocumentRequest) -> bool:
        snapshot = self._snapshot()
        return bool(
            self.editor.is_attached
            and snapshot is not None
            and replace(
                snapshot, key=replace(snapshot.key, request_id=request.key.request_id)
            )
            == request
            and self.editor.text == request.text
            and self.request == request
        )

    def synchronize(self, *, immediate: bool = False) -> None:
        """Retire prior status immediately; schedule only the current source."""
        if self.retired:
            return
        snapshot = self._snapshot()
        editable = self._editable()
        if snapshot is None:
            self.retire()
            return
        if self._was_editable and not editable:
            self.retire()
            return
        self._was_editable = editable
        if (
            self.request is not None
            and replace(
                snapshot,
                key=replace(snapshot.key, request_id=self.request.key.request_id),
            )
            == self.request
            and not immediate
        ):
            return
        language_changed = (
            self.request is None or self.request.language != snapshot.language
        )
        self._request_id += 1
        self.request = replace(
            snapshot, key=replace(snapshot.key, request_id=self._request_id)
        )
        self.result = None
        self._pending_immediate = immediate or language_changed
        self._publish(
            self.request,
            None,
            "Checking" if immediate or language_changed else "Pending",
            False,
        )
        if self._lease is not None:
            self._lease.submit(self.request, immediate=immediate or language_changed)
        elif self._acquiring is None:
            self._generation += 1
            self._acquiring = asyncio.create_task(self._acquire(self._generation))

    async def _acquire(self, generation: int) -> None:
        if generation != self._generation or self.request is None:
            return
        try:
            lease = await self._worker.acquire(self._receive)
        except RuntimeError:
            if generation == self._generation and self.request is not None:
                self._acquiring = None
                self._receive(
                    CheckResult.for_request(
                        self.request, "failed", False, (), False, False
                    )
                )
            return
        if generation != self._generation or self.request is None:
            lease.retire()
            return
        self._acquiring = None
        self._lease = lease
        lease.submit(self.request, immediate=self._pending_immediate)

    def _receive(self, result: CheckResult) -> None:
        request = self.request
        if request is None or result.key != request.key or not self._current(request):
            return
        self.result = result
        self._publish(request, result, "", self._editable())

    def check(self) -> None:
        """Manual checking uses the same bounded lane and immutable snapshot."""
        self.synchronize(immediate=True)

    async def format(self) -> bool:
        """Apply a qualified proposal once through the widget's native history."""
        request = self.request
        if (
            request is None
            or self._lease is None
            or self.result is None
            or not self.result.format_allowed
            or not self._editable()
            or not self._current(request)
        ):
            return False
        self.bridge.capture_request(request)
        generation = self._generation
        result = await self._lease.format(request)
        if (
            result is None
            or generation != self._generation
            or not self._current(request)
        ):
            return False
        if self._map_format is not None:
            result = self._map_format(result, request)
        applied = self.bridge.apply_format(result, request.key, self._editable())
        if applied:
            self.editor.focus()
        elif result.state == "unchanged":
            self._publish(request, result.check, "Already formatted", self._editable())
        else:
            self._receive(result.check)
        return applied

    def navigate(self, index: int) -> bool:
        """Focus a current diagnostic at logical Unicode source coordinates."""
        request, result = self.request, self.result
        if (
            request is None
            or result is None
            or not self._current(request)
            or not 0 <= index < len(result.diagnostics)
        ):
            return False
        offset = result.diagnostics[index].start
        self.editor.move_cursor(
            cast(Document, self.editor.document).get_location_from_index(offset)
        )
        self.editor.focus()
        return True

    def retire(self) -> None:
        """Permanently retire losing authority, including pending acquisition."""
        self.retired = True
        self._generation += 1
        if self._lease is not None:
            self._lease.retire()
            self._lease = None
        self._acquiring = None
        self.request = None
        self.result = None
        self._was_editable = False
        # A newer capture also revokes any bridge proposal held by a caller.
        self.bridge = TextAreaFormatBridge(self.editor)
        self._publish(None, None, "Inactive", False)
