"""Application-owned Canvas browser authority for the native Console."""

from __future__ import annotations

import json
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
    CanvasBridgeSettlementLease,
    CanvasGatewayEvent,
    CanvasGatewayNavigation,
    CanvasGatewayOption,
    CanvasGatewayProjection,
    CanvasGatewayScope,
    CanvasSourceResponse,
)
from .limits import validate_utf8_text
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


class NativeConsoleCanvasAuthority:
    """Resolve all browser state from the exact active Console scope."""

    def __init__(
        self,
        *,
        scope_resolver: Callable[[str], CanvasScope],
        canvas_controller: Any,
        bridge_sink: Callable[[str], None] | None = None,
        auto_open: Callable[[str, CanvasRevisionInfo], None] | None = None,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._canvas_controller = canvas_controller
        self._bridge_sink = bridge_sink
        self._auto_open = auto_open
        self._selection: dict[str, _Selection] = {}
        self._events: dict[tuple[str, str], list[CanvasGatewayEvent]] = {}
        self._lock = RLock()

    def rebind_view(
        self,
        *,
        scope_resolver: Callable[[str], CanvasScope],
        bridge_sink: Callable[[str], None] | None,
        auto_open: Callable[[str, CanvasRevisionInfo], None] | None,
    ) -> None:
        """Refresh view callbacks while preserving runtime-owned selections."""

        self._scope_resolver = scope_resolver
        self._bridge_sink = bridge_sink
        self._auto_open = auto_open

    def on_tool_mutation(self, scope: CanvasScope, result: Any) -> None:
        """Publish successful tool updates and request first-view auto-open."""

        info = getattr(result, "revision", None)
        if not isinstance(info, CanvasRevisionInfo):
            return
        with self._lock:
            previous = self._selection.get(scope.session_id)
            self._selection[scope.session_id] = _Selection(
                info.canvas_id, info.revision_id, True
            )
            self._publish(
                scope.session_id,
                info,
                "updated" if previous is not None else "selection_changed",
            )
        if self._auto_open is not None:
            self._auto_open(scope.session_id, info)

    def import_html(
        self, *, session_id: str, source: str, create_new: bool = False
    ) -> CanvasRevisionInfo:
        """Create a revision from an authorized transcript block and select it."""

        scope = self._scope_resolver(session_id)
        title = _document_title(source)
        with self._lock:
            selection = self._selection.get(session_id)
            if not create_new and selection is not None:
                result = self._update(
                    scope, selection.canvas_id, selection.revision_id, source
                )
                info = result.revision
            else:
                created = self._create(scope, title, source)
                info = created.revision
            self._selection[session_id] = _Selection(info.canvas_id, info.revision_id)
            self._publish(session_id, info, "selection_changed")
            return info

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
            current = self._read_exact(canvas_scope, scope.canvas_id, scope.revision_id)
            following = self._selection.get(
                session_id, _Selection(scope.canvas_id, scope.revision_id)
            ).following
            if action == "select":
                if not canvas_id:
                    raise ValueError("Canvas selection is required")
                current = self._choose(canvas_scope, session_id, canvas_id, None)
                following = True
            elif action == "pin":
                following = False
            elif action == "follow":
                current = self._choose(canvas_scope, session_id, scope.canvas_id, None)
                following = True
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
    ) -> BridgeConfirmationResponse:
        if (
            not request.approved
            or request.request.kind != "submit"
            or self._bridge_sink is None
        ):
            return BridgeConfirmationResponse(request.request.request_id, "cancelled")
        text = _draft_payload(request.request.value)
        if settlement.try_settle(lambda: self._bridge_sink(text)):
            return BridgeConfirmationResponse(request.request.request_id, "confirmed")
        return BridgeConfirmationResponse(request.request.request_id, "refused")

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

    def _create(self, scope: CanvasScope, title: str, source: str) -> Any:
        result = self._canvas_controller.interactive_create_canvas(
            scope,
            origin_message_id=scope.active_message_ids[-1],
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

    def _update(self, scope: CanvasScope, canvas_id: str, parent_id: str, source: str):
        result = self._canvas_controller.interactive_update_canvas(
            scope,
            origin_message_id=scope.active_message_ids[-1],
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
