"""Application-owned Canvas browser authority for the native Console."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from threading import RLock
from typing import Any
from uuid import uuid4

from .compiler import compile_canvas_document
from .gateway import (
    BridgeConfirmationRequest,
    BridgeConfirmationResponse,
    BridgePreparationResponse,
    CanvasBridgeSettlementLease,
    CanvasGatewayEvent,
    CanvasGatewayNavigation,
    CanvasGatewayOption,
    CanvasGatewayProjection,
    CanvasGatewayScope,
    CanvasSourceResponse,
)
from .limits import sha256_utf8, validate_utf8_text
from .models import (
    CanvasReadResult,
    CanvasRevisionInfo,
    CanvasScope,
)


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.parts.append(data)


def _document_title(source: str) -> str:
    parser = _TitleParser()
    parser.feed(source)
    title = " ".join("".join(parser.parts).split())
    return title[:200] or "Canvas"


def _validated_title(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Canvas title must be text")
    title = " ".join(value.split())
    validate_utf8_text(title, limit=800, field_name="Canvas title")
    if not title or len(title) > 200:
        raise ValueError("Canvas title must be 1–200 characters")
    return title


def _draft_payload(value: Any) -> str:
    """Preserve text verbatim and serialize structured bridge values as JSON."""

    if isinstance(value, str):
        return value

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [thaw(child) for child in item]
        return item

    return json.dumps(
        thaw(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(slots=True)
class _Selection:
    canvas_id: str
    revision_id: str
    following: bool = True


@dataclass(frozen=True, slots=True)
class _ParsedBlockImport:
    canvas_id: str
    revision_id: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class CanvasBridgeTarget:
    """Exact source-free native chat target captured by one browser shell."""

    browser_session_id: str
    session_id: str
    conversation_id: str
    active_message_ids: tuple[str, ...]
    canvas_id: str
    revision_id: str


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedNativeBridge:
    """Ephemeral composer guard or passive-download approval without payload data."""

    target: CanvasBridgeTarget
    request_id: str
    kind: str
    apply_submit: Callable[[str], None] | None = None

    def __repr__(self) -> str:
        return (
            "_PreparedNativeBridge("
            f"request_id={self.request_id!r}, kind={self.kind!r}, payload=<absent>)"
        )


@dataclass(slots=True)
class _PublicationReceipt:
    state_published: bool = False
    auto_opened: bool = False
    auto_open_in_flight: bool = False


_MAX_PARSED_BLOCK_IMPORTS = 512
_MAX_BROWSER_BINDINGS = 64
_MAX_PUBLICATION_RECEIPTS = 256


class NativeConsoleCanvasAuthority:
    """Resolve all browser state from the exact active Console scope."""

    def __init__(
        self,
        *,
        scope_resolver: Callable[[str], CanvasScope],
        canvas_controller: Any,
        bridge_sink: Callable[[CanvasBridgeTarget, str], None] | None = None,
        bridge_prepare: Callable[[CanvasBridgeTarget], Callable[[str], None]] | None = None,
        auto_open: Callable[[str, CanvasRevisionInfo], None] | None = None,
        publication_guard: Callable[[Any], bool] | None = None,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._canvas_controller = canvas_controller
        self._bridge_sink = bridge_sink
        self._bridge_prepare = bridge_prepare
        self._auto_open = auto_open
        self._publication_guard = publication_guard
        self._gateway_invalidator: Callable[[str], None] | None = None
        self._selection: dict[str, _Selection] = {}
        self._parsed_block_imports: OrderedDict[
            tuple[str, str, int], _ParsedBlockImport
        ] = OrderedDict()
        self._browser_targets: OrderedDict[str, CanvasBridgeTarget] = OrderedDict()
        self._publication_receipts: OrderedDict[str, _PublicationReceipt] = (
            OrderedDict()
        )
        self._events: dict[tuple[str, str], list[CanvasGatewayEvent]] = {}
        self._lock = RLock()
        self._disposed = False

    def dispose(self) -> None:
        """Fence publication replay state and release runtime-owned memory."""

        with self._lock:
            self._disposed = True
            self._publication_receipts.clear()
            self._parsed_block_imports.clear()
            self._browser_targets.clear()
            self._selection.clear()
            self._events.clear()
            self._gateway_invalidator = None

    def bind_gateway_invalidator(
        self, invalidator: Callable[[str], None] | None
    ) -> None:
        """Bind the runtime-owned capability revoker for unreachable shells."""

        self._gateway_invalidator = invalidator

    def rebind_view(
        self,
        *,
        scope_resolver: Callable[[str], CanvasScope],
        bridge_sink: Callable[[CanvasBridgeTarget, str], None] | None,
        bridge_prepare: Callable[[CanvasBridgeTarget], Callable[[str], None]] | None = None,
        auto_open: Callable[[str, CanvasRevisionInfo], None] | None,
        publication_guard: Callable[[Any], bool] | None = None,
    ) -> None:
        """Refresh view callbacks while preserving runtime-owned selections."""

        self._scope_resolver = scope_resolver
        self._bridge_sink = bridge_sink
        self._bridge_prepare = bridge_prepare
        self._auto_open = auto_open
        self._publication_guard = publication_guard

    def on_settlement_publication(self, publication: Any) -> None:
        """Publish committed tool revisions that still belong to the live branch."""

        scope = getattr(publication, "scope", None)
        revisions = getattr(publication, "revisions", ())
        publication_id = getattr(publication, "publication_id", None)
        if (
            not isinstance(scope, CanvasScope)
            or not revisions
            or not isinstance(publication_id, str)
            or not publication_id.startswith("publication-")
        ):
            return
        if self._publication_guard is not None:
            if not self._publication_guard(publication):
                return
        else:
            current = self._scope_resolver(scope.session_id)
            if (
                current.session_id != scope.session_id
                or current.conversation_id != scope.conversation_id
                or any(
                    not isinstance(info, CanvasRevisionInfo)
                    or info.origin.message_id not in current.active_message_ids
                    for info in revisions
                )
            ):
                return
        info = revisions[-1]
        with self._lock:
            if self._disposed:
                raise RuntimeError("canvas_publication_authority_disposed")
            receipt = self._publication_receipts.get(publication_id)
            if receipt is None:
                if len(self._publication_receipts) >= _MAX_PUBLICATION_RECEIPTS:
                    delivered = next(
                        (
                            receipt_id
                            for receipt_id, candidate in (
                                self._publication_receipts.items()
                            )
                            if candidate.state_published
                            and candidate.auto_opened
                            and not candidate.auto_open_in_flight
                        ),
                        None,
                    )
                    if delivered is None:
                        raise RuntimeError(
                            "canvas_publication_capacity_exhausted"
                        )
                    # The controller marks its listener delivered only after
                    # this method completes, so fully completed receipts are
                    # the sole replay-safe eviction candidates.
                    self._publication_receipts.pop(delivered, None)
                receipt = _PublicationReceipt()
                self._publication_receipts[publication_id] = receipt
            if not receipt.state_published:
                previous = self._selection.get(scope.session_id)
                self._selection[scope.session_id] = _Selection(
                    info.canvas_id, info.revision_id, True
                )
                for revision in revisions:
                    self._publish(
                        scope.session_id,
                        revision,
                        "updated"
                        if previous is not None
                        and previous.canvas_id == revision.canvas_id
                        else "selection_changed",
                    )
                    previous = _Selection(revision.canvas_id, revision.revision_id)
                receipt.state_published = True
            if receipt.auto_opened:
                return
            if receipt.auto_open_in_flight:
                raise RuntimeError("canvas_publication_already_opening")
            receipt.auto_open_in_flight = True
        try:
            if self._auto_open is not None:
                self._auto_open(scope.session_id, info)
        except Exception:
            with self._lock:
                receipt.auto_open_in_flight = False
            raise
        with self._lock:
            receipt.auto_open_in_flight = False
            receipt.auto_opened = True

    def import_html(
        self,
        *,
        session_id: str,
        source: str,
        create_new: bool = False,
        source_message_id: str | None = None,
        origin_message_id: str | None = None,
        source_turn_id: str | None = None,
        block_index: int | None = None,
        block_identity: str | None = None,
    ) -> CanvasRevisionInfo:
        """Create a revision from an authorized transcript block and select it."""

        scope = self._scope_resolver(session_id)
        title = _document_title(source)
        parsed_key = self._parsed_block_key(
            scope,
            source_message_id=source_message_id,
            origin_message_id=origin_message_id,
            source_turn_id=source_turn_id,
            block_index=block_index,
            block_identity=block_identity,
        )
        revision_origin_message_id = (
            origin_message_id or source_message_id or scope.active_message_ids[-1]
        )
        origin_turn_id = source_turn_id or scope.run_id
        with self._lock:
            if parsed_key is not None and not create_new:
                prior = self._parsed_block_imports.get(parsed_key)
                if prior is None:
                    persisted = self._canvas_controller.find_interactive_import(
                        scope,
                        origin_message_id=revision_origin_message_id,
                        origin_turn_id=origin_turn_id,
                        content_sha256=sha256_utf8(source),
                        temporary=self._is_temporary(scope),
                    )
                    if persisted is not None:
                        prior = _ParsedBlockImport(
                            persisted.canvas_id,
                            persisted.revision_id,
                            persisted.content_sha256,
                        )
                        self._parsed_block_imports[parsed_key] = prior
                if prior is not None:
                    if prior.content_sha256 != sha256_utf8(source):
                        raise RuntimeError("Canvas block identity changed")
                    selected_scope = replace(
                        scope,
                        selected_canvas_id=prior.canvas_id,
                        selected_revision_id=prior.revision_id,
                    )
                    existing = self._read_exact(
                        selected_scope, prior.canvas_id, prior.revision_id
                    )
                    self._parsed_block_imports.move_to_end(parsed_key)
                    self._selection[session_id] = _Selection(
                        existing.revision.canvas_id,
                        existing.revision.revision_id,
                    )
                    return existing.revision
            selection = self._selection.get(session_id)
            if not create_new and selection is not None:
                result = self._update(
                    scope,
                    selection.canvas_id,
                    selection.revision_id,
                    source,
                    origin_message_id=revision_origin_message_id,
                    origin_turn_id=origin_turn_id,
                )
                info = result.revision
            else:
                created = self._create(
                    scope,
                    title,
                    source,
                    origin_message_id=revision_origin_message_id,
                    origin_turn_id=origin_turn_id,
                )
                info = created.revision
            if parsed_key is not None and not create_new:
                self._parsed_block_imports[parsed_key] = _ParsedBlockImport(
                    info.canvas_id, info.revision_id, info.content_sha256
                )
                self._parsed_block_imports.move_to_end(parsed_key)
                while len(self._parsed_block_imports) > _MAX_PARSED_BLOCK_IMPORTS:
                    self._parsed_block_imports.popitem(last=False)
            self._selection[session_id] = _Selection(info.canvas_id, info.revision_id)
            self._publish(session_id, info, "selection_changed")
            return info

    @staticmethod
    def _parsed_block_key(
        scope: CanvasScope,
        *,
        source_message_id: str | None,
        origin_message_id: str | None,
        source_turn_id: str | None,
        block_index: int | None,
        block_identity: str | None,
    ) -> tuple[str, str, int] | None:
        values = (
            source_message_id,
            origin_message_id,
            source_turn_id,
            block_index,
            block_identity,
        )
        if all(value is None for value in values):
            return None
        if (
            not isinstance(source_message_id, str)
            or not source_message_id
            or not isinstance(origin_message_id, str)
            or not origin_message_id
            or origin_message_id not in scope.active_message_ids
        ):
            raise RuntimeError("Canvas source message is not on the active branch")
        if not isinstance(source_turn_id, str) or not source_turn_id:
            raise ValueError("Canvas source turn is invalid")
        if type(block_index) is not int or not 0 <= block_index <= 1024:
            raise ValueError("Canvas block index is invalid")
        if block_identity != f"{source_message_id}:canvas-html:{block_index}":
            raise ValueError("Canvas block identity is invalid")
        return (scope.conversation_id, origin_message_id, block_index)

    def gateway_scope(
        self,
        *,
        session_id: str,
        browser_session_id: str,
        canvas_id: str | None = None,
        revision_id: str | None = None,
        follow_latest: bool = True,
    ) -> CanvasGatewayScope:
        """Select a reachable Canvas and return its source-free browser scope."""

        scope = self._scope_resolver(session_id)
        with self._lock:
            chosen = self._choose(scope, session_id, canvas_id, revision_id)
            self._selection[session_id] = _Selection(
                chosen.revision.canvas_id,
                chosen.revision.revision_id,
                following=follow_latest,
            )
            event_key = (session_id, chosen.revision.canvas_id)
            prior_events = self._events.get(event_key)
            if prior_events and prior_events[-1].kind == "disconnected":
                prior_events.pop()
                if not prior_events:
                    self._events.pop(event_key, None)
            self._browser_targets[browser_session_id] = CanvasBridgeTarget(
                browser_session_id=browser_session_id,
                session_id=session_id,
                conversation_id=scope.conversation_id,
                active_message_ids=scope.active_message_ids,
                canvas_id=chosen.revision.canvas_id,
                revision_id=chosen.revision.revision_id,
            )
            self._browser_targets.move_to_end(browser_session_id)
            while len(self._browser_targets) > _MAX_BROWSER_BINDINGS:
                self._browser_targets.popitem(last=False)
            return CanvasGatewayScope(
                browser_session_id=browser_session_id,
                conversation_session_id=session_id,
                canvas_id=chosen.revision.canvas_id,
                revision_id=chosen.revision.revision_id,
            )

    def resolve_render_plan(self, scope: CanvasGatewayScope):
        return compile_canvas_document(self._read_gateway(scope).source)

    def read_source(self, scope: CanvasGatewayScope) -> CanvasSourceResponse:
        read = self._read_gateway(scope)
        return CanvasSourceResponse(read.source, read.revision.content_sha256)

    def describe_selection(self, scope: CanvasGatewayScope) -> CanvasGatewayProjection:
        canvas_scope = self._selected_scope(scope)
        displayed = self._read_exact(canvas_scope, scope.canvas_id, scope.revision_id)
        items = self._list(
            replace(
                canvas_scope,
                selected_canvas_id=None,
                selected_revision_id=None,
            )
        )
        selection = self._selection.get(scope.conversation_session_id)
        return CanvasGatewayProjection(
            scope=scope,
            options=tuple(
                CanvasGatewayOption(item.canvas_id, item.revision_id, item.title)
                for item in items
            ),
            title=displayed.revision.title,
            sequence=displayed.revision.sequence,
            parent_revision_id=displayed.revision.parent_revision_id,
            source_bytes=displayed.revision.source_bytes,
            content_sha256=displayed.revision.content_sha256,
            origin_message_id=displayed.revision.origin.message_id,
            origin_turn_id=displayed.revision.origin.run_id,
            temporary=self._is_temporary(canvas_scope),
            following=selection.following if selection is not None else True,
        )

    def navigate(
        self,
        scope: CanvasGatewayScope,
        *,
        action: str,
        canvas_id: str | None = None,
        title: str | None = None,
    ) -> CanvasGatewayNavigation:
        """Apply one server-owned select/pin/follow/previous/rename transition."""

        session_id = scope.conversation_session_id
        canvas_scope = self._selected_scope(scope)
        with self._lock:
            following = self._selection.get(
                session_id, _Selection(scope.canvas_id, scope.revision_id)
            ).following
            if action == "select":
                if not canvas_id:
                    raise ValueError("Canvas selection is required")
                current = self._choose(canvas_scope, session_id, canvas_id, None)
                following = True
            elif action == "follow":
                current = self._choose(canvas_scope, session_id, scope.canvas_id, None)
                following = True
            else:
                current = self._read_exact(
                    canvas_scope, scope.canvas_id, scope.revision_id
                )
                if action == "pin":
                    following = False
                elif action == "previous":
                    parent = current.revision.parent_revision_id
                    if parent is None:
                        raise ValueError("Previous revision is unavailable")
                    current = self._read_exact(canvas_scope, scope.canvas_id, parent)
                    following = False
                elif action == "rename":
                    validated = _validated_title(title if title is not None else "")
                    current = self._rename(canvas_scope, current, validated)
                else:
                    raise ValueError("Unsupported Canvas navigation")
            self._selection[session_id] = _Selection(
                current.revision.canvas_id,
                current.revision.revision_id,
                following,
            )
            next_scope = replace(
                scope,
                canvas_id=current.revision.canvas_id,
                revision_id=current.revision.revision_id,
            )
            return CanvasGatewayNavigation(
                next_scope, self.describe_selection(next_scope)
            )

    def sync_live_context(self, session_id: str | None) -> None:
        """Publish the reachable head after a real session or branch transition."""

        invalidated: list[str] = []
        with self._lock:
            for stale_session_id in tuple(self._selection):
                if stale_session_id != session_id:
                    invalidated.extend(
                        self._invalidate_unreachable_selection(stale_session_id)
                    )
            if session_id is None or self._selection.get(session_id) is None:
                self._revoke_browser_targets(invalidated)
                return
        scope = self._scope_resolver(session_id)
        with self._lock:
            selection = self._selection.get(session_id)
            if selection is None:
                return
            items = self._list(
                replace(scope, selected_canvas_id=None, selected_revision_id=None)
            )
            item = next(
                (
                    candidate
                    for candidate in items
                    if candidate.canvas_id == selection.canvas_id
                ),
                None,
            )
            selected_reachable = True
            try:
                self._read_exact(
                    scope, selection.canvas_id, selection.revision_id
                )
            except (RuntimeError, ValueError):
                selected_reachable = False
            if item is None or (not selection.following and not selected_reachable):
                invalidated.extend(
                    self._invalidate_unreachable_selection(session_id)
                )
                self._revoke_browser_targets(invalidated)
                return
            if item.revision_id == selection.revision_id:
                self._revoke_browser_targets(invalidated)
                return
            reachable = self._read_exact(scope, item.canvas_id, item.revision_id)
            if selection.following:
                self._selection[session_id] = _Selection(
                    item.canvas_id, item.revision_id, True
                )
            self._publish(session_id, reachable.revision, "selection_changed")
        self._revoke_browser_targets(invalidated)

    def _invalidate_unreachable_selection(self, session_id: str) -> list[str]:
        """Drop one stale selection and publish a source-free terminal event."""

        selection = self._selection.pop(session_id, None)
        if selection is None:
            return []
        browser_ids = [
            browser_id
            for browser_id, target in self._browser_targets.items()
            if target.session_id == session_id
            and target.canvas_id == selection.canvas_id
        ]
        for browser_id in browser_ids:
            self._browser_targets.pop(browser_id, None)
        self._events.setdefault((session_id, selection.canvas_id), []).append(
            CanvasGatewayEvent(
                f"event-{uuid4().hex}",
                "disconnected",
                selection.canvas_id,
                selection.revision_id,
                {"notice": "unavailable_on_branch"},
            )
        )
        return browser_ids

    def _revoke_browser_targets(self, browser_ids: list[str]) -> None:
        invalidator = self._gateway_invalidator
        if invalidator is None:
            return
        for browser_id in browser_ids:
            invalidator(browser_id)

    def read_events(
        self, scope: CanvasGatewayScope, *, after_event_id: str | None
    ) -> tuple[CanvasGatewayEvent, ...]:
        events = tuple(
            self._events.get((scope.conversation_session_id, scope.canvas_id), ())
        )
        if after_event_id is None:
            return events[-1:]
        for index, event in enumerate(events):
            if event.event_id == after_event_id:
                return events[index + 1 :]
        return events[-1:]

    def confirm_bridge(
        self,
        scope: CanvasGatewayScope,
        request: BridgeConfirmationRequest,
        *,
        settlement: CanvasBridgeSettlementLease,
        preparation: object | None = None,
    ) -> BridgeConfirmationResponse:
        if not request.approved:
            return BridgeConfirmationResponse(request.request.request_id, "cancelled")
        if (
            not isinstance(preparation, _PreparedNativeBridge)
            or preparation.request_id != request.request.request_id
            or preparation.kind != request.request.kind
        ):
            return BridgeConfirmationResponse(request.request.request_id, "refused")
        target = preparation.target
        if not self._bridge_target_is_current(scope, target):
            return BridgeConfirmationResponse(request.request.request_id, "refused")
        try:
            if request.request.kind == "submit":
                if preparation.apply_submit is None:
                    return BridgeConfirmationResponse(request.request.request_id, "cancelled")
                text = request.request.submit_text()
                effect = lambda: self._apply_bridge_effect(
                    scope, target, text, preparation.apply_submit
                )
            else:
                request.request.download_payload()
                effect = lambda: self._apply_passive_download_approval(scope, target)
            if settlement.try_settle(effect):
                return BridgeConfirmationResponse(request.request.request_id, "confirmed")
        except RuntimeError:
            pass
        return BridgeConfirmationResponse(request.request.request_id, "refused")

    def prepare_bridge(
        self,
        scope: CanvasGatewayScope,
        request: Any,
    ) -> tuple[BridgePreparationResponse, object]:
        """Capture the exact composer before trusted UI exposes a confirmation."""

        from .models import CanvasBridgeRequest

        if not isinstance(request, CanvasBridgeRequest):
            raise TypeError("Canvas bridge preparation requires a validated request")
        target = self._browser_targets.get(scope.browser_session_id)
        if target is None or not self._bridge_target_is_current(scope, target):
            raise RuntimeError("Canvas bridge target is unavailable")
        projection = self.describe_selection(scope)
        if projection.scope != scope:
            raise RuntimeError("Canvas bridge target is unavailable")
        if request.kind == "submit":
            apply_submit = None
            if self._bridge_prepare is not None:
                apply_submit = self._bridge_prepare(target)
            elif self._bridge_sink is not None:
                apply_submit = lambda text: self._bridge_sink(target, text)
            complete_text = request.submit_text()
            presentation = BridgePreparationResponse(
                request_id=request.request_id,
                kind=request.kind,
                conversation_id=target.conversation_id,
                canvas_id=target.canvas_id,
                revision_id=target.revision_id,
                canvas_title=projection.title,
                revision_number=projection.sequence,
                complete_text=complete_text,
                byte_size=len(complete_text.encode("utf-8")),
            )
        else:
            download = request.download_payload()
            apply_submit = None
            presentation = BridgePreparationResponse(
                request_id=request.request_id,
                kind=request.kind,
                conversation_id=target.conversation_id,
                canvas_id=target.canvas_id,
                revision_id=target.revision_id,
                canvas_title=projection.title,
                revision_number=projection.sequence,
                complete_text=download.text_preview,
                filename=download.filename,
                mime_type=download.mime_type,
                byte_size=len(download.data),
            )
        return presentation, _PreparedNativeBridge(
            target=target,
            request_id=request.request_id,
            kind=request.kind,
            apply_submit=apply_submit,
        )

    def _apply_bridge_effect(
        self,
        scope: CanvasGatewayScope,
        target: CanvasBridgeTarget,
        text: str,
        apply_submit: Callable[[str], None],
    ) -> None:
        """Revalidate inside settlement immediately before the composer effect."""

        if not self._bridge_target_is_current(scope, target):
            raise RuntimeError("Canvas bridge target is unavailable")
        apply_submit(text)

    def _apply_passive_download_approval(
        self, scope: CanvasGatewayScope, target: CanvasBridgeTarget
    ) -> None:
        """Linearize browser-owned passive download approval against live scope."""

        if not self._bridge_target_is_current(scope, target):
            raise RuntimeError("Canvas bridge target is unavailable")

    def _bridge_target_is_current(
        self, scope: CanvasGatewayScope, target: CanvasBridgeTarget
    ) -> bool:
        if (
            target.browser_session_id != scope.browser_session_id
            or target.session_id != scope.conversation_session_id
            or target.canvas_id != scope.canvas_id
            or target.revision_id != scope.revision_id
        ):
            return False
        try:
            current = self._scope_resolver(target.session_id)
        except RuntimeError:
            return False
        return (
            current.session_id == target.session_id
            and current.conversation_id == target.conversation_id
            and current.active_message_ids == target.active_message_ids
        )

    def _selected_scope(self, gateway_scope: CanvasGatewayScope) -> CanvasScope:
        scope = self._scope_resolver(gateway_scope.conversation_session_id)
        return replace(
            scope,
            selected_canvas_id=gateway_scope.canvas_id,
            selected_revision_id=gateway_scope.revision_id,
        )

    def _read_gateway(self, scope: CanvasGatewayScope) -> CanvasReadResult:
        return self._read_exact(
            self._selected_scope(scope), scope.canvas_id, scope.revision_id
        )

    def _create(
        self,
        scope: CanvasScope,
        title: str,
        source: str,
        *,
        origin_message_id: str,
        origin_turn_id: str,
    ) -> Any:
        result = self._canvas_controller.interactive_create_canvas(
            scope,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
            title=title,
            html=source,
            temporary=self._is_temporary(scope),
        )
        return type(
            "InteractiveCreate",
            (),
            {
                "revision": result.revision,
                "source": source,
                "compatibility_issues": result.compatibility_issues,
            },
        )()

    def _update(
        self,
        scope: CanvasScope,
        canvas_id: str,
        parent_id: str,
        source: str,
        *,
        origin_message_id: str,
        origin_turn_id: str,
    ):
        result = self._canvas_controller.interactive_update_canvas(
            scope,
            origin_message_id=origin_message_id,
            origin_turn_id=origin_turn_id,
            canvas_id=canvas_id,
            expected_parent_revision_id=parent_id,
            html=source,
            temporary=self._is_temporary(scope),
        )
        if not hasattr(result, "revision"):
            raise RuntimeError("Canvas changed before import")
        return result

    def _rename(
        self, scope: CanvasScope, current: CanvasReadResult, title: str
    ) -> CanvasReadResult:
        result = self._canvas_controller.interactive_rename_canvas(
            scope,
            origin_message_id=scope.active_message_ids[-1],
            canvas_id=current.revision.canvas_id,
            expected_parent_revision_id=current.revision.revision_id,
            title=title,
            temporary=self._is_temporary(scope),
        )
        if not hasattr(result, "revision"):
            raise RuntimeError("Canvas changed before rename")
        return self._read_exact(
            replace(
                scope,
                selected_canvas_id=result.revision.canvas_id,
                selected_revision_id=result.revision.revision_id,
            ),
            result.revision.canvas_id,
            result.revision.revision_id,
        )

    def _list(self, scope: CanvasScope):
        return self._canvas_controller.list_session_canvases(
            scope, temporary=self._is_temporary(scope)
        )

    def _choose(
        self,
        scope: CanvasScope,
        session_id: str,
        canvas_id: str | None,
        revision_id: str | None,
    ) -> CanvasReadResult:
        if canvas_id is None:
            selected = self._selection.get(session_id)
            canvas_id = selected.canvas_id if selected else None
        items = self._list(
            replace(scope, selected_canvas_id=None, selected_revision_id=None)
        )
        if canvas_id is None and items:
            canvas_id = items[0].canvas_id
        if canvas_id is None:
            raise ValueError("No Canvas is available")
        if revision_id is None:
            item = next((item for item in items if item.canvas_id == canvas_id), None)
            if item is None:
                raise ValueError("Canvas is not reachable")
            revision_id = item.revision_id
        return self._read_exact(scope, canvas_id, revision_id)

    def _read_exact(
        self, scope: CanvasScope, canvas_id: str, revision_id: str
    ) -> CanvasReadResult:
        selected = replace(
            scope, selected_canvas_id=canvas_id, selected_revision_id=revision_id
        )
        return self._canvas_controller.read_session_canvas(
            selected, canvas_id, temporary=self._is_temporary(scope)
        )

    def _publish(self, session_id: str, info: CanvasRevisionInfo, kind: str) -> None:
        metadata = {
            "title": info.title,
            "sequence": info.sequence,
            "source_bytes": info.source_bytes,
            "content_sha256": info.content_sha256,
            "temporary": self._is_temporary(self._scope_resolver(session_id)),
            "origin_message_id": info.origin.message_id,
            "origin_turn_id": info.origin.run_id,
        }
        self._events.setdefault((session_id, info.canvas_id), []).append(
            CanvasGatewayEvent(
                f"event-{uuid4().hex}", kind, info.canvas_id, info.revision_id, metadata
            )
        )

    def _is_temporary(self, scope: CanvasScope) -> bool:
        return scope.conversation_id == scope.session_id
