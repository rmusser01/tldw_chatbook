"""Bounded, inert Console reads over the retained native owners (ADR-244)."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import TYPE_CHECKING

from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBatch

from .console_remote_commands import REVISION_FIELDS, ConsoleRemoteCommands
from .console_remote_models import ConsoleHostIdentity, ConsoleTarget

if TYPE_CHECKING:
    from tldw_chatbook.Web_Server.served_console_events import ConsoleEventBuffer

    from .console_runtime import ConsoleRuntime

PAGE_BYTES = 64 * 1024
LATEST_MESSAGE_BYTES = 8 * 1024
LATEST_MESSAGE_CHARS = 2000
MAX_CLIENTS = 64


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True).encode("utf-8")


class ConsoleRemoteProjection:
    """Project native state without constructing an engine or changing selection."""

    def __init__(
        self,
        runtime: ConsoleRuntime,
        host_identity: ConsoleHostIdentity,
        events: ConsoleEventBuffer,
    ) -> None:
        self.runtime = runtime
        self.host_identity = host_identity
        self.events = events
        self._commands = ConsoleRemoteCommands(runtime, host_identity)
        self._last_change_token = None
        self._captured_boundary = None
        self._clients: set[str] = set()
        self._answerable_clients: dict[ConsoleTarget, set[str]] = {}

    def _require_target(self, target: ConsoleTarget):
        store, controller, session = self._commands._resolve(target)
        if (
            target.turn_id is not None
            and controller._current_control_turn_id(session.id) != target.turn_id
        ):
            raise ValueError("Console target turn changed.")
        if target.round_id is not None:
            if target.handle_id is not None or not any(
                row["round_id"] == target.round_id and row["run_id"] == target.run_id
                for row in controller._interrupt_host.remote_rounds(session.id)
            ):
                raise ValueError("Console target decision changed.")
        elif target.handle_id is not None:
            bridge = self.runtime.agent_bridge
            if (
                bridge is None
                or not session.persisted_conversation_id
                or not target.run_id
                or not any(
                    handle.handle_id == target.handle_id
                    and handle.run_id == target.run_id
                    for handle in bridge.fleet_snapshot(
                        session.persisted_conversation_id
                    )
                )
            ):
                raise ValueError("Console target worker changed.")
        elif (
            target.run_id is not None
            and controller._current_primary_run_id(session.id) != target.run_id
        ):
            raise ValueError("Console target run changed.")
        return store, controller, session

    def register_client(self, client_id: str) -> None:
        """Register a parent-issued authenticated live subscription, never a device row."""
        from .console_turn_preparation import _validate_identifier

        _validate_identifier(client_id, "subscription ID")
        if client_id not in self._clients and len(self._clients) >= MAX_CLIENTS:
            raise ValueError("Console subscription capacity reached.")
        self._clients.add(client_id)

    def disconnect_client(self, client_id: str) -> None:
        """Release only this live subscription's exact rendered-round claims."""
        self._clients.discard(client_id)
        for target, clients in tuple(self._answerable_clients.items()):
            clients.discard(client_id)
            if not clients:
                self._answerable_clients.pop(target)
                self._publish_round_answerability(target, False)

    def close(self) -> None:
        """Retire projection claims without cancelling native work."""
        for client_id in tuple(self._clients):
            self.disconnect_client(client_id)

    def set_answerable(
        self, client_id: str, target: ConsoleTarget, *, answerable: bool
    ) -> None:
        """Mark a supported exact round rendered by a currently authenticated client."""
        _store, controller, _session = self._require_target(target)
        if client_id not in self._clients or type(answerable) is not bool:
            raise ValueError("An authenticated live subscription is required.")
        host = controller._interrupt_host
        row = next(
            (
                row
                for row in host.remote_rounds(target.session_id)
                if row["round_id"] == target.round_id and row["run_id"] == target.run_id
            ),
            None,
        )
        if row is None or not row["supported"]:
            raise ValueError("Decision is no longer remotely answerable.")
        self._prune_answerability()
        if (
            target not in self._answerable_clients
            and len(self._answerable_clients) >= 100
        ):
            raise ValueError("Console rendered-decision capacity reached.")
        clients = self._answerable_clients.setdefault(target, set())
        if answerable:
            clients.add(client_id)
        else:
            clients.discard(client_id)
        self._publish_round_answerability(target, bool(clients))
        if not clients:
            self._answerable_clients.pop(target, None)

    def _publish_round_answerability(
        self, target: ConsoleTarget, answerable: bool
    ) -> None:
        controller = self.runtime.chat_controller
        if controller is not None:
            controller._interrupt_host.set_remote_answerable(
                target, answerable=answerable
            )

    def subscribe(self, after_sequence: int) -> ConsoleEventBatch:
        """Read bounded independent invalidations, retaining loss until reconciliation."""
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("Invalid Console event cursor.")
        return self.events.read(after_sequence)

    def acknowledge_snapshot(self, sequence: int) -> None:
        """Acknowledge only a completed native capture, never a parent transport cursor."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Invalid native snapshot boundary.")
        if sequence == self._captured_boundary:
            self.events.acknowledge_snapshot(sequence)

    def _native_token(self) -> str:
        """Read revision witnesses through native owners, without widget state."""
        store, controller = self.runtime.chat_store, self.runtime.chat_controller
        tokens = []
        for session in store.sessions() if store is not None else ():
            tokens.append(
                (
                    session.id,
                    session.incarnation_id,
                    session.title,
                    session.workspace_id,
                    session.updated_at,
                    getattr(controller, "_active_assistant_message_ids", {}).get(
                        session.id
                    )
                    if controller
                    else None,
                    store.display_projection_revision(session.id),
                    store.session_fingerprint(session.id),
                    tuple(getattr(session, key) for key in REVISION_FIELDS),
                    controller._lifecycle_revision_for(session.id)
                    if controller
                    else None,
                    controller._interrupt_host.remote_rounds(session.id)
                    if controller
                    else (),
                    self.runtime.remote_project_decisions(session.id),
                    tuple(
                        (
                            handle.handle_id,
                            handle.run_id,
                            handle.status,
                            getattr(handle, "queued_steering", 0),
                            getattr(handle, "undelivered_steering", 0),
                            str(getattr(handle, "task", ""))[:200],
                            str(getattr(handle, "result", ""))[:1000],
                            str(getattr(handle, "error", ""))[:1000],
                        )
                        for handle in self.runtime.agent_bridge.fleet_snapshot(
                            session.persisted_conversation_id
                        )
                    )
                    if self.runtime.agent_bridge is not None
                    and session.persisted_conversation_id
                    else (),
                )
            )
        registry = self._workspace_registry()
        tokens.append(getattr(registry, "mutation_generation", None))
        return hashlib.sha256(_encoded(tokens)).hexdigest()

    def poll_changes(self) -> int | None:
        """Read native revision witnesses at the existing publisher cadence."""
        token = self._native_token()
        self._prune_answerability()
        if token == self._last_change_token:
            return None
        self._last_change_token = token
        sequence = self.events.publish("console.changed", {})
        self.events.note_changed(sequence)
        return sequence

    def _prune_answerability(self) -> None:
        controller = self.runtime.chat_controller
        if controller is None:
            self.close()
            return
        for target in tuple(self._answerable_clients):
            rows = controller._interrupt_host.remote_rounds(target.session_id)
            if not any(
                row["round_id"] == target.round_id and row["run_id"] == target.run_id
                for row in rows
            ):
                self._answerable_clients.pop(target, None)
                self._publish_round_answerability(target, False)

    def _workspace_registry(self):
        # Read only already-wired services; a lazy app getter can construct stores.
        ready = vars(self.runtime.app)
        return ready.get("workspace_registry_service") or ready.get(
            "_workspace_registry_service"
        )

    def _chat_metadata(self, workspace_id, last_modified) -> dict:
        from tldw_chatbook.Canvas.limits import validate_opaque_identifier
        from tldw_chatbook.Workspaces.registry_service import (
            WorkspaceRegistryServiceError,
        )

        workspace = workspace_id if workspace_id not in (None, "global") else None
        if workspace is not None:
            workspace = validate_opaque_identifier(workspace, field_name="workspace ID")
        name = None
        registry = self._workspace_registry()
        if workspace is not None and registry is not None:
            try:
                record = registry.get_workspace(workspace)
                name = str(record.name)[:160] if record is not None else None
            except (ValueError, RuntimeError, WorkspaceRegistryServiceError):
                pass  # Missing/unreadable names remain unavailable, never guessed.
        return {
            "scope_type": "workspace" if workspace is not None else "global",
            "workspace_id": workspace,
            "workspace_name": name,
            "last_modified": str(last_modified)[:64]
            if last_modified is not None
            else None,
        }

    def _discovery_rows(self, window: int) -> tuple[list, int | None]:
        rows = self._rows(None) if window == 0 else []
        store = self.runtime.chat_store
        live = (
            {session.persisted_conversation_id for session in store.sessions()}
            if store
            else set()
        )
        service = getattr(self.runtime.app, "local_chat_conversation_service", None)
        if service is None:
            return rows, None
        saved = service.list_library_conversations(
            limit=100, offset=window, archive_scope="all"
        )
        rows.extend(
            (
                "chats",
                {
                    "conversation_id": row["id"],
                    "title": str(row["title"])[:160],
                    "live": False,
                    **self._chat_metadata(
                        row.get("workspace_id"), row.get("last_modified")
                    ),
                },
            )
            for row in saved["items"]
            if row["id"] not in live
        )
        following = window + len(saved["items"])
        return rows, following if saved["items"] and following < saved[
            "total"
        ] else None

    def _rows(self, target: ConsoleTarget | None) -> list[tuple[str, dict]]:
        runtime = self.runtime
        store, controller = runtime.chat_store, runtime.chat_controller
        if target is not None:
            store, controller, session = self._require_target(target)
            sessions = [session]
        else:
            sessions = store.sessions() if store is not None else []
        rows = []
        for session in sessions:
            if not store.saved_session_ready(session):
                continue
            scope = dict(
                self.host_identity.model_dump(),
                session_id=session.id,
                incarnation_id=session.incarnation_id,
            )
            state = controller.run_state_for(session.id) if controller else None
            status = getattr(state, "status", None)
            rows.append(
                (
                    "chats",
                    dict(
                        scope,
                        conversation_id=session.persisted_conversation_id,
                        title=str(session.title)[:160],
                        pending_decision_count=min(
                            100,
                            len(
                                controller._interrupt_host.remote_rounds(session.id)
                                if controller
                                else ()
                            )
                            + len(runtime.remote_project_decisions(session.id)),
                        ),
                        live=True,
                        **self._chat_metadata(session.workspace_id, session.updated_at),
                        status=getattr(status, "value", status),
                        run_id=controller._current_primary_run_id(session.id)
                        if controller
                        else None,
                        turn_id=controller._current_control_turn_id(session.id)
                        if controller
                        else None,
                        expected_revisions={
                            key: getattr(session, key) for key in REVISION_FIELDS
                        },
                    ),
                )
            )
            if target is None:
                continue
            bridge = runtime.agent_bridge
            if bridge is not None and session.persisted_conversation_id:
                for handle in bridge.fleet_snapshot(session.persisted_conversation_id):
                    rows.append(
                        (
                            "fleet",
                            dict(
                                scope,
                                handle_id=handle.handle_id,
                                run_id=handle.run_id,
                                status=handle.status,
                                live=handle.status
                                not in {
                                    "done",
                                    "error",
                                    "stuck",
                                    "cancelled",
                                    "superseded",
                                },
                                agent=str(getattr(handle, "agent", None) or "Worker")[
                                    :160
                                ],
                                task=str(getattr(handle, "task", ""))[:200],
                                activity=str(
                                    getattr(handle, "error", "")
                                    or getattr(handle, "result", "")
                                )[:1000],
                                queued_steering=max(
                                    0,
                                    min(1000000, getattr(handle, "queued_steering", 0)),
                                ),
                                undelivered_steering=max(
                                    0,
                                    min(
                                        1000000,
                                        getattr(handle, "undelivered_steering", 0),
                                    ),
                                ),
                                started_at=getattr(handle, "started_at", None),
                                finished_at=getattr(handle, "finished_at", None),
                            ),
                        )
                    )
            if controller is not None:
                for row in controller._interrupt_host.remote_rounds(session.id):
                    rows.append(("decisions", dict(scope, **row)))
            for row in runtime.remote_project_decisions(session.id):
                rows.append(("decisions", dict(scope, **row)))
            recovery = store.dispatch_recovery_for_presentation(session.id)
            if recovery is not None:
                rows.append(
                    ("recoveries", dict(scope, pending=True, action="open_terminal"))
                )
            for message in store.read_only_messages_for_session(session.id):
                if (
                    message.role not in {"user", "assistant", "tool"}
                    or message.generation_projection_quarantined
                ):
                    continue
                rows.append(
                    (
                        "messages",
                        dict(
                            scope,
                            message_id=message.id,
                            role=message.role,
                            status=message.status,
                            content=message.content,
                            offset=0,
                        ),
                    )
                )
        return rows

    def _latest_message(self, rows: list, target: ConsoleTarget) -> dict | None:
        messages = [row for section, row in rows if section == "messages"]
        if not messages:
            return None
        active = getattr(
            self.runtime.chat_controller, "_active_assistant_message_ids", {}
        ).get(target.session_id)
        original = next(
            (
                row
                for row in messages
                if row["message_id"] == active and row["role"] == "assistant"
            ),
            messages[-1],
        )
        content = original["content"]
        preview = dict(
            original, content="", offset=len(content), total_chars=len(content)
        )
        if len(_encoded(preview)) > LATEST_MESSAGE_BYTES:
            return None
        low, high = 0, min(LATEST_MESSAGE_CHARS, len(content))
        while low < high:
            length = (low + high + 1) // 2
            preview.update(content=content[-length:], offset=len(content) - length)
            if len(_encoded(preview)) <= LATEST_MESSAGE_BYTES:
                low = length
            else:
                high = length - 1
        preview.update(content=content[-low:] if low else "", offset=len(content) - low)
        return preview

    def _receipt_fields(self, cursor: str | None) -> dict:
        receipts = self.runtime._remote_receipts
        result = {"command_receipts": [], "next_receipt_cursor": None}
        if receipts is None:
            return result
        page = receipts.repository.list_unresolved(cursor=cursor, limit=20)
        for receipt in page.items:
            row = receipt.public()
            # Reserve most of the existing frame budget for native projection.
            if len(_encoded([*result["command_receipts"], row])) > 16 * 1024:
                break
            result["command_receipts"].append(row)
        if len(result["command_receipts"]) < len(page.items):
            result["next_receipt_cursor"] = result["command_receipts"][-1]["command_id"]
        else:
            result["next_receipt_cursor"] = page.next_cursor
        return result

    @staticmethod
    def _continue_receipts(page: dict, receipt_cursor: str | None) -> None:
        # A keyset end remains resumable while native content is still paging.
        if page["next_cursor"] is not None and page["next_receipt_cursor"] is None:
            page["next_receipt_cursor"] = (
                page["command_receipts"][-1]["command_id"]
                if page["command_receipts"]
                else receipt_cursor
            )

    def _complete_capture(self, page: dict, token: str) -> dict:
        if self._native_token() != token:
            sequence = self.events.publish("console.changed", {"reset_required": True})
            self.events.note_changed(sequence)
            page["reset_required"] = True
        elif page["next_cursor"] is None and page["next_receipt_cursor"] is None:
            self._captured_boundary = page["sequence"]
        return page

    def _saved_cursor(self, target: ConsoleTarget) -> str | None:
        store, _controller, session = self._require_target(target)
        if (
            getattr(self.runtime.app, "local_chat_conversation_service", None) is None
            or not session.persisted_conversation_id
            or store.message_count(session.id)
        ):
            return None
        scope = dict(
            target.model_dump(), conversation_id=session.persisted_conversation_id
        )
        return base64.urlsafe_b64encode(
            _encoded(
                {
                    "saved": True,
                    "scope": hashlib.sha256(
                        _encoded([scope, self._rows(target)])
                    ).hexdigest(),
                    "offset": 0,
                    "char_start": 0,
                    "revision": None,
                }
            )
        ).decode()

    def _saved_snapshot(self, target, cursor, boundary, receipt_fields, receipt_cursor):
        """Continue canonical history after native metadata; never hydrate to read."""
        _store, _controller, session = self._require_target(target)
        service = getattr(self.runtime.app, "local_chat_conversation_service", None)
        current = self._saved_cursor(target)
        if current is None:
            return None
        scope_revision = json.loads(base64.urlsafe_b64decode(current))["scope"]
        offset, char_start, expected = 0, 0, None
        reset = boundary.reset_required
        if cursor is not None:
            try:
                if not isinstance(cursor, str) or len(cursor) > 2048:
                    raise ValueError
                prior = json.loads(base64.urlsafe_b64decode(cursor))
                if set(prior) != {"saved", "scope", "offset", "char_start", "revision"}:
                    raise ValueError
                if prior["scope"] != scope_revision:
                    return None
                elif (
                    type(prior["offset"]) is int
                    and 0 <= prior["offset"] <= 1000000
                    and type(prior["char_start"]) is int
                    and 0 <= prior["char_start"] <= 100000000
                ):
                    offset, char_start, expected = (
                        prior["offset"],
                        prior["char_start"],
                        prior["revision"],
                    )
                else:
                    raise ValueError
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("Invalid saved-history cursor.") from error
        saved = service.get_library_conversation_messages(
            session.persisted_conversation_id,
            message_offset=offset,
            message_limit=1,
            max_chars=2000,
            char_start=char_start,
        )
        if saved is None:
            raise ValueError("Saved conversation is no longer available.")
        revision = f"{saved['version']}:{saved['message_epoch']}"
        if expected is not None and expected != revision:
            return None  # Restart with native metadata, not a partial history reset.
        page = dict(
            self.host_identity.model_dump(),
            sequence=boundary.sequence,
            reset_required=reset,
            **receipt_fields,
            chats=[],
            messages=[],
            fleet=[],
            decisions=[],
            recoveries=[],
            next_cursor=None,
        )
        if saved["messages"]:
            message = saved["messages"][0]
            if message["sender"] in {"user", "assistant", "tool"}:
                page["messages"].append(
                    {
                        "message_id": message["id"],
                        "role": message["sender"],
                        "content": message["text"],
                        "offset": char_start,
                        "complete": not message["has_more"],
                        "revision": message["revision"],
                        "session_id": session.id,
                    }
                )
            if message["has_more"]:
                char_start += message["returned_chars"]
            else:
                offset, char_start = offset + 1, 0
        if (
            char_start
            or offset < saved["message_total"]
            or page["next_receipt_cursor"] is not None
        ):
            page["next_cursor"] = base64.urlsafe_b64encode(
                _encoded(
                    {
                        "saved": True,
                        "scope": scope_revision,
                        "offset": offset,
                        "char_start": char_start,
                        "revision": revision,
                    }
                )
            ).decode()
        self._continue_receipts(page, receipt_cursor)
        self._require_target(target)
        return page

    def snapshot(
        self,
        target: ConsoleTarget | None,
        *,
        cursor: str | None = None,
        limit: int = 50,
        receipt_cursor: str | None = None,
    ) -> dict:
        """Capture after registering a cursor; never acknowledge a concurrent change."""
        if self.runtime._disposed:
            raise RuntimeError("Console owner is shutting down.")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Console page limit must be between 1 and 100.")
        if target is not None:
            self._require_target(target)
        # This pre-capture boundary remains the answer even when native capture publishes.
        boundary = self.events.read(self.events.read(0).sequence)
        token = self._native_token()
        receipt_fields = self._receipt_fields(receipt_cursor)
        cursor_reset = False
        if target is not None and cursor is not None:
            try:
                if not isinstance(cursor, str) or len(cursor) > 2048:
                    raise ValueError
                candidate = json.loads(base64.urlsafe_b64decode(cursor))
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid Console page cursor.") from error
            if isinstance(candidate, dict) and "saved" in candidate:
                saved = self._saved_snapshot(
                    target, cursor, boundary, receipt_fields, receipt_cursor
                )
                if saved is not None:
                    return self._complete_capture(saved, token)
                cursor, cursor_reset = None, True
        prior = None
        if cursor is not None:
            if not isinstance(cursor, str) or len(cursor) > 2048:
                raise ValueError("Invalid Console page cursor.")
            try:
                prior = json.loads(base64.urlsafe_b64decode(cursor))
                if (
                    type(prior) is not dict
                    or set(prior) != {"revision", "index", "offset", "window"}
                    or any(
                        type(prior[key]) is not int or prior[key] < 0
                        for key in ("index", "offset", "window")
                    )
                    or (
                        prior["revision"] is not None
                        and not isinstance(prior["revision"], str)
                    )
                    or prior["window"] > 1000000
                    or (target is not None and prior["window"])
                ):
                    raise ValueError
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError("Invalid Console page cursor.") from error
        window = prior["window"] if prior else 0
        if target is None:
            rows, next_window = self._discovery_rows(window)
        else:
            rows, next_window = self._rows(target), None
        scope = (
            self.host_identity.model_dump() if target is None else target.model_dump()
        )
        revision = hashlib.sha256(_encoded([scope, rows])).hexdigest()
        index, offset = 0, 0
        reset = boundary.reset_required or cursor_reset
        if prior is not None:
            if prior["revision"] == revision or (
                prior["revision"] is None and target is None
            ):
                index, offset = prior["index"], prior["offset"]
                if index > len(rows) or (
                    offset
                    and (
                        index == len(rows)
                        or rows[index][0] != "messages"
                        or offset > len(rows[index][1]["content"])
                    )
                ):
                    raise ValueError("Invalid Console page cursor.")
            else:
                reset = True
        page = dict(
            self.host_identity.model_dump(),
            sequence=boundary.sequence,
            reset_required=reset,
            **receipt_fields,
            chats=[],
            messages=[],
            next_cursor=None,
            fleet=[],
            decisions=[],
            recoveries=[],
        )
        if target is not None:
            page["latest_message"] = self._latest_message(rows, target)
        count = 0
        while index < len(rows) and count < limit:
            section, original = rows[index]
            row = dict(original)
            if section == "messages":
                content = original["content"]
                low, high = 0, len(content) - offset
                while low < high:
                    length = (low + high + 1) // 2
                    row.update(content=content[offset : offset + length], offset=offset)
                    page[section].append(row)
                    fits = len(_encoded(page)) <= PAGE_BYTES - 544
                    page[section].pop()
                    if fits:
                        low = length
                    else:
                        high = length - 1
                if low == 0 and content[offset:]:
                    break
                row.update(
                    content=content[offset : offset + low],
                    offset=offset,
                    complete=offset + low == len(content),
                )
            page[section].append(row)
            if len(_encoded(page)) > PAGE_BYTES - 512:
                page[section].pop()
                break
            count += 1
            if section == "messages" and not row["complete"]:
                offset += len(row["content"])
                break
            index, offset = index + 1, 0
        if index < len(rows):
            page["next_cursor"] = base64.urlsafe_b64encode(
                _encoded(
                    {
                        "revision": revision,
                        "index": index,
                        "offset": offset,
                        "window": window,
                    }
                )
            ).decode()
        elif next_window is not None:
            page["next_cursor"] = base64.urlsafe_b64encode(
                _encoded(
                    {"revision": None, "index": 0, "offset": 0, "window": next_window}
                )
            ).decode()
        if target is not None:
            self._require_target(target)
            if page["next_cursor"] is None:
                page["next_cursor"] = self._saved_cursor(target)
        if page["next_cursor"] is None and page["next_receipt_cursor"] is not None:
            page["next_cursor"] = base64.urlsafe_b64encode(
                _encoded(
                    {
                        "revision": revision,
                        "index": index,
                        "offset": offset,
                        "window": window,
                    }
                )
            ).decode()
        self._continue_receipts(page, receipt_cursor)
        # Reading a page is not an acknowledgement of an entire paginated snapshot.
        return self._complete_capture(page, token)
