"""Lazy redacted pages from one captured approval; never an authority input."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING

from tldw_chatbook.MCP.redaction import redact_mapping

if TYPE_CHECKING:
    from tldw_chatbook.Chat.approval_presentation import ApprovalRowView

DetailsIdentity = tuple[str, int, int, str]
_UNAVAILABLE = "Arguments unavailable"
_CACHE_PAGES = 2


@dataclass(frozen=True)
class ApprovalDetailsPage:
    """One bounded literal page and an honest continuation flag."""

    index: int
    text: str
    has_more: bool


def iter_redacted_details(
    argument_sets: Sequence[Mapping[str, object]],
    *,
    page_chars: int = 4096,
    targets: Sequence[str] = (),
) -> Iterator[ApprovalDetailsPage]:
    """Encode all captured sets incrementally after display-only redaction.

    Args:
        argument_sets: Original captured inputs, never mutated.
        page_chars: Maximum characters in each displayed page.
        targets: Complete already-redacted captured display targets; when supplied,
            prepend a target record without changing the default argument-list shape.

    Yields:
        Complete consecutive pages of one JSON array, or a safe error page.
    """
    if page_chars <= 0:
        raise ValueError("Invalid page size")
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2)

    def chunks() -> Iterator[str]:
        yield "["
        if targets:
            yield from encoder.iterencode({"captured_targets": targets})
        for index, arguments in enumerate(argument_sets):
            if index or targets:
                yield ","
            yield from encoder.iterencode(redact_mapping(dict(arguments)))
        yield "]"

    buffer = ""
    index = 0
    try:
        for chunk in chunks():
            offset = 0
            while offset < len(chunk):
                count = min(page_chars + 1 - len(buffer), len(chunk) - offset)
                buffer += chunk[offset : offset + count]
                offset += count
                if len(buffer) > page_chars:
                    yield ApprovalDetailsPage(index, buffer[:page_chars], True)
                    index += 1
                    buffer = buffer[page_chars:]
        yield ApprovalDetailsPage(index, buffer, False)
    except Exception:
        yield ApprovalDetailsPage(index, _UNAVAILABLE, False)


class ApprovalDetailsController:
    """Prepare captured pages off the UI thread and fence their delivery."""

    def __init__(
        self,
        *,
        current_identity: Callable[[], DetailsIdentity | None],
        spawn_worker: Callable[[Callable[[], None]], object],
        post_page: Callable[[DetailsIdentity, ApprovalDetailsPage], None],
        paint_page: Callable[[ApprovalDetailsPage], None],
        paint_loading: Callable[[], None],
    ) -> None:
        self._current_identity = current_identity
        self._spawn_worker = spawn_worker
        self._post_page = post_page
        self._paint_page = paint_page
        self._paint_loading = paint_loading
        self._identity: DetailsIdentity | None = None
        self._arguments: Sequence[Mapping[str, object]] = ()
        self._targets: Sequence[str] = ()
        self._requested = 0
        self._cancel = Event()
        self._pages: OrderedDict[int, ApprovalDetailsPage] = OrderedDict()

    def open(
        self, row: ApprovalRowView, *, round_id: str, revision: int, generation: int
    ) -> None:
        """Open one captured row without doing serialization on the UI thread.

        Args:
            row: Immutable owner snapshot.
            round_id: Owning approval round.
            revision: Semantic snapshot revision.
            generation: Current card gesture generation.
        """
        self.close()
        self._identity = (round_id, revision, generation, row.verdict_key)
        self._arguments = row.argument_sets
        self._targets = row.targets
        self.request_page(0)

    def request_page(self, index: int) -> None:
        """Prepare or paint an available page for the current captured row."""
        identity = self._identity
        if identity is None or index < 0:
            return
        self._requested = index
        self._cancel.set()
        self._cancel = Event()
        cancel = self._cancel
        cached = self._pages.get(index)
        if cached is not None:
            self.deliver_page(identity, cached)
            return
        self._paint_loading()
        arguments = self._arguments
        targets = self._targets
        post_page = self._post_page

        # ponytail: replay from the capture after two-page cache eviction; retain a
        # worker-owned iterator if measured navigation cost warrants more state.
        def prepare() -> None:
            if cancel.is_set():
                return
            for page in iter_redacted_details(arguments, targets=targets):
                if cancel.is_set():
                    return
                if page.index == index:
                    try:
                        post_page(identity, page)
                    except Exception:
                        pass
                    return
                if not page.has_more:
                    return

        self._spawn_worker(prepare)

    def deliver_page(
        self, identity: DetailsIdentity, page: ApprovalDetailsPage
    ) -> bool:
        """Paint only the current identity and latest requested page on UI thread."""
        if (
            identity != self._identity
            or identity != self._current_identity()
            or page.index != self._requested
            or self._cancel.is_set()
        ):
            return False
        self._pages[page.index] = page
        self._pages.move_to_end(page.index)
        while len(self._pages) > _CACHE_PAGES:
            self._pages.popitem(last=False)
        self._paint_page(page)
        return True

    def close(self) -> None:
        """Cancel cooperatively and release every prepared page/capture reference."""
        self._cancel.set()
        self._identity = None
        self._arguments = ()
        self._targets = ()
        self._pages.clear()
