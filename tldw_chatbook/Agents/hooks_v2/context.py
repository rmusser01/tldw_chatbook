"""Host-attributed context owned by a live session and its exact input scopes."""

from __future__ import annotations

import json
from collections.abc import Callable
from threading import RLock

from ..agent_models import HookContextOrigin, PluginContextText, check_host_context
from .engine import HookEventOutcome
from .models import ContextBlock, HookEvent


class ContextLedger:
    """Keep ephemeral contributions separate from durable summary/history rows."""

    def __init__(
        self,
        *,
        owners=lambda owner: (owner,),
        render_context=None,
        effects_current=lambda *_: True,
        lock=None,
    ):
        self._owners = owners
        self._render = render_context or self.render
        self._effects_current = effects_current
        self._accepted = {}
        self._lock = lock if lock is not None else RLock()
        self._events = {}
        self._rows = {}
        self._delivered = {}
        self._nested = {}

    @staticmethod
    def render(event: HookEvent, handler_id: str, block: ContextBlock) -> str:
        return (
            "<untrusted-hook-context>\n"
            + json.dumps(
                {
                    "event_id": event.event_id,
                    "handler_id": handler_id,
                    "instructions": block.text,
                },
                ensure_ascii=False,
            )
            + "\n</untrusted-hook-context>"
        )

    def bind(
        self,
        owner_id: str,
        event: HookEvent,
        *,
        current: Callable[[], bool] = lambda: True,
    ) -> None:
        with self._lock:
            self._events[event.event_id] = (owner_id, current)

    def accept(self, event: HookEvent, result: HookEventOutcome) -> None:
        """Stage complete attributed blocks; the checkpoint lock owns publication."""
        with self._lock:
            owner, current = self._events[event.event_id]
            if not current():
                raise ValueError("hook context owner stale")
            rows = []
            for handler_id, effects in result.accepted:
                for block in effects.context:
                    rendered = self._render(event, handler_id, block)
                    origins = (
                        rendered.checked_origins()
                        if isinstance(rendered, PluginContextText)
                        else ()
                    )
                    origin = HookContextOrigin(
                        event.event_id,
                        handler_id,
                        len(rendered.encode("utf-8")),
                        origins,
                    )
                    rows.append(
                        {
                            "role": "user",
                            "content": PluginContextText(
                                str(rendered), origins, (origin,)
                            ),
                        }
                    )
            existing = [
                row
                for key, value in self._rows.items()
                if self._events[key][0] in self._owners(owner)
                for row in value
            ]
            check_host_context(existing + rows, strip=False)
            self._rows[event.event_id] = tuple(rows)
            self._accepted[event.event_id] = (event, result)
            try:
                self.commit_nested(event, result, current=current)
            except BaseException:
                self._rows.pop(event.event_id, None)
                self._accepted.pop(event.event_id, None)
                raise

    def stage_nested(
        self, container, handler_id, owner, event, result, rows, *, current
    ):
        """Stage an internal tool event until its enclosing handler is accepted."""
        with self._lock:
            self._owners(owner)  # Positively require the exact live input scope.
            if not current() or not self._effects_current(event, result):
                raise ValueError("nested hook context stale")
            children = self._selected_nested(event, result)
            batch = [(handler_id, owner, event, result, tuple(rows), current)]
            for (
                _handler,
                child_owner,
                child_event,
                child_result,
                child_rows,
                child_current,
            ) in children:
                if child_owner != owner:
                    raise ValueError("nested hook input owner changed")

                def probe(previous=child_current):
                    return (
                        current()
                        and previous()
                        and self._effects_current(event, result)
                    )

                batch.append(
                    (handler_id, owner, child_event, child_result, child_rows, probe)
                )
            existing = [
                row
                for key, value in self._rows.items()
                if self._events[key][0] in self._owners(owner)
                for row in value
            ]
            pending = [
                row
                for values in self._nested.values()
                for item in values
                if item[1] == owner
                for row in item[4]
            ]
            # Children move, they are not counted twice in the pending batch.
            moving = [
                row for item in self._nested.get(event.event_id, ()) for row in item[4]
            ]
            if moving:
                for row in moving:
                    pending.remove(row)
            check_host_context(
                existing + pending + [row for item in batch for row in item[4]],
                strip=False,
            )
            self._nested.pop(event.event_id, None)
            self._nested.setdefault(container.event_id, []).extend(batch)

    def _selected_nested(self, event, result):
        if not result.allowed or result.outstanding_cleanup:
            return ()
        accepted = {handler for handler, _value in result.accepted}
        return tuple(
            item for item in self._nested.get(event.event_id, ()) if item[0] in accepted
        )

    def commit_nested(self, event, result, *, current=lambda: True, additional_rows=()):
        """Publish a complete accepted nested batch into its exact input scope."""
        with self._lock:
            batch = self._selected_nested(event, result)
            owners = {owner for _handler, owner, *_rest in batch}
            ancestry = {
                ancestor for owner in owners for ancestor in self._owners(owner)
            }
            existing = [
                row
                for key, rows in self._rows.items()
                if self._events[key][0] in ancestry
                for row in rows
            ]
            for _handler, _owner, nested_event, nested_result, _rows, probe in batch:
                if (
                    not current()
                    or not probe()
                    or not self._effects_current(nested_event, nested_result)
                ):
                    raise ValueError("nested hook context stale")
            check_host_context(
                existing
                + list(additional_rows)
                + [row for item in batch for row in item[4]],
                strip=False,
            )
            for _handler, owner, nested_event, nested_result, rows, probe in batch:

                def accepted_current(previous=probe):
                    return (
                        current()
                        and previous()
                        and self._effects_current(event, result)
                    )

                key = nested_event.event_id
                self._events[key] = (owner, accepted_current)
                self._rows[key] = rows
                self._accepted[key] = (nested_event, nested_result)
            self._nested.pop(event.event_id, None)

    def blocks(self, owner_id: str, boundary: str) -> tuple[dict, ...]:
        """Deliver each contribution once per exact receiving input owner."""
        with self._lock:
            owners = self._owners(owner_id)
            seen = self._delivered.setdefault((owner_id, boundary), set())
            rows = []
            for key, contribution in self._rows.items():
                owner, current = self._events[key]
                if owner not in owners:
                    continue
                if not current() or not self._effects_current(*self._accepted[key]):
                    raise ValueError("accepted hook context became stale")
                if key in seen:
                    continue
                rows.extend(dict(row) for row in contribution)
                seen.add(key)
            return tuple(rows)

    def transfer(self, source: str, target: str, *, current) -> None:
        """Consume one manually committed operation's context into a real turn."""
        with self._lock:
            for key, (owner, _previous) in tuple(self._events.items()):
                if owner == source:
                    self._events[key] = (target, current)

    def close(self, owner_id: str) -> None:
        with self._lock:
            for key, batch in tuple(self._nested.items()):
                kept = [item for item in batch if item[1] != owner_id]
                if kept:
                    self._nested[key] = kept
                else:
                    self._nested.pop(key, None)
            for key, (owner, _current) in tuple(self._events.items()):
                if owner == owner_id:
                    self._events.pop(key, None)
                    self._rows.pop(key, None)
                    self._accepted.pop(key, None)
            for key in tuple(self._delivered):
                if key[0] == owner_id:
                    self._delivered.pop(key, None)
