"""Stable read-only observers of the selected client's actual publications.

These objects own no controller, store or database. Compatibility accessors
expose published values and one explicitly requested original history read.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
from dataclasses import dataclass
from types import FunctionType, MethodType

from .console_execution_projection_values import ConsoleSessionView
from .console_execution_projection_view import (
    ConsoleProjectionView,
    ConsoleProjectionUnavailable,
)


@dataclass(frozen=True, slots=True)
class ConsoleSessionReader:
    owner: ConsoleStoreReader
    session_id: str
    incarnation: str

    def _workspace_facts(self):
        from .console_execution_workspace_values import ConsoleWorkspaceSurface

        view = self.owner._current()
        if view.session(self.session_id).incarnation_id != self.incarnation:
            raise ConsoleProjectionUnavailable(self.session_id)
        surface = self.owner._client.workspace_surface
        if type(surface) is ConsoleWorkspaceSurface:
            for facts in surface.sessions:
                if facts.session_id == self.session_id:
                    return facts
        raise ConsoleProjectionUnavailable(self.session_id)

    def local_character_id(self) -> int | None:
        return self._workspace_facts().character_id

    @property
    def character_name(self) -> str | None:
        return self._workspace_facts().character_name

    def __getattr__(self, name):
        if name not in ConsoleSessionView.__dataclass_fields__:
            raise AttributeError(name)
        row = self.owner._current().session(self.session_id)
        if row.incarnation_id != self.incarnation:
            raise ConsoleProjectionUnavailable(self.session_id)
        return getattr(row, name)


class ConsoleStoreReader:
    """Keep reader identity while each call observes one coherent publication."""

    def __init__(self, client):
        self._client = client
        self._sessions = {}

    def _current(self) -> ConsoleProjectionView:
        from .console_execution_process import ConsoleExecutionClient

        require_reader_sources()
        client = self._client
        if (
            type(client) is not ConsoleExecutionClient
            or client.store_reader is not self
            or client._app.console_execution_client is not client
            or client._disposed
            or client._channel_error is not None
        ):
            raise ConsoleProjectionUnavailable("Console reader owner changed")
        view = client.projection_view
        if (
            type(view) is not ConsoleProjectionView
            or view.source_revision != client._source_revision
        ):
            raise ConsoleProjectionUnavailable("Console publication is unavailable")
        return view

    @property
    def active_session_id(self):
        return self._current().active_session_id

    def sessions(self):
        view = self._current()
        keys = {(row.id, row.incarnation_id) for row in view.session_rows}
        self._sessions = {
            key: item for key, item in self._sessions.items() if key in keys
        }
        return [self._session(row) for row in view.session_rows]

    def _session(self, row):
        key = row.id, row.incarnation_id
        if key not in self._sessions:
            self._sessions[key] = ConsoleSessionReader(self, *key)
        return self._sessions[key]

    def session(self, session_id):
        return self._session(self._current().session(session_id))

    def messages_for_session(self, session_id):
        return self._current().messages_for_session(session_id)

    def has_messages(self, session_id: str) -> bool:
        """Read the original transcript count from the current publication."""
        return self.session_fingerprint(session_id)[0] > 0

    def pending_attachments(self, session_id):
        """Return detached staged values only for this publication's active row."""
        from .attachment_core import PendingAttachment
        from .console_execution_protocol import _PROJECTED_INPUTS, _detach

        view = self._current()
        row = view.session(session_id)
        inputs = _detach(self._client.pressed_inputs, "pressed_inputs", 0, set())
        _PROJECTED_INPUTS(inputs, "pressed_inputs")
        if (
            view.active_session_id != session_id
            or inputs["session_id"] != session_id
            or inputs["incarnation_id"] != row.incarnation_id
            or inputs["settings_revision"] != row.settings_revision
            or tuple(item["attachment_id"] for item in inputs["attachments"])
            != inputs["attachment_ids"]
        ):
            raise ConsoleProjectionUnavailable("Console attachment publication changed")
        return [
            PendingAttachment(
                attachment_id=item["attachment_id"], file_path=item["file_path"],
                display_name=item["display_name"], file_type=item["file_type"],
                insert_mode=item["insert_mode"],
                data=None if item["data"] is None else base64.b64decode(item["data"], validate=True),
                mime_type=item["mime_type"], text_content=item["text_content"],
                original_size=item["original_size"], processed_size=item["processed_size"],
            )
            for item in inputs["attachments"]
        ]

    def session_fingerprint(self, session_id):
        return self._current().session_fingerprint(session_id)

    def display_projection_revision(self, session_id):
        return self._current().display_projection_revision(session_id)


class ConsoleRunHistoryReader:
    """Request original AgentRunsDB rows without exposing a database handle."""

    def __init__(self, client):
        self._client = client
        self._read_history = _definition(type(client).read_run_history)

    @property
    def _db(self):
        # This compatibility name denotes this reader, never an AgentRunsDB.
        return self

    def list_runs(
        self, conversation_id, include_superseded=True, limit=None, agent_kind=None
    ):
        client = self._client
        client.store_reader._current()
        reader = client.read_run_history
        if (
            client.controller_reader._agent_bridge is not self
            or type(reader) is not MethodType
            or reader.__self__ is not client
            or reader.__func__ is not self._read_history[0]
            or not _definition_current(self._read_history)
        ):
            raise ConsoleProjectionUnavailable("Console history reader changed")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Console history observation must leave the owner loop")
        request = {
            "conversation_id": conversation_id,
            "include_superseded": include_superseded,
            "limit": limit,
            "agent_kind": agent_kind,
        }
        return asyncio.run_coroutine_threadsafe(
            reader(request), client._loop
        ).result()


class ConsoleControllerReader:
    """Original UI read names backed solely by authenticated presentation facts."""

    def __init__(self, client):
        self._client = client
        self._agent_bridge = ConsoleRunHistoryReader(client)

    @property
    def store(self):
        self._client.store_reader._current()
        if self._client.controller_reader is not self:
            raise ConsoleProjectionUnavailable("Console controller reader changed")
        return self._client.store_reader

    @property
    def run_state(self):
        """Read the original active-session run state from this publication."""
        view = self.store._current()
        return view._projection_for_session(view.active_session_id).run_state

    @property
    def _parked_approval_payloads(self):
        view = self.store._current()
        frame = view._projection_for_session(view.active_session_id)
        decision = frame.decision
        return (
            decision
            if decision is not None and decision.decision_type == "approval"
            else None
        )

    def _head_round_payload(self, payloads, session_id):
        current = self._parked_approval_payloads
        if payloads is not current:
            raise ConsoleProjectionUnavailable("Console approval publication changed")
        if current is None or current.session_id != session_id:
            return None
        return current.payload


from .console_domain_services import _definition, _definition_current

_READERS = tuple(
    (owner, name, value, _definition(value.fget if type(value) is property else value))
    for owner in (
        ConsoleSessionReader,
        ConsoleStoreReader,
        ConsoleRunHistoryReader,
        ConsoleControllerReader,
    )
    for name, value in vars(owner).items()
    if type(value) in (FunctionType, property)
)
_VIEW_READERS = tuple(
    (name, _definition(getattr(ConsoleProjectionView, name)))
    for name in (
        "session",
        "messages_for_session",
        "session_fingerprint",
        "display_projection_revision",
        "_projection_for_session",
    )
)


def require_reader_sources():
    for owner, name, original, record in _READERS:
        declared = type.__getattribute__(owner, "__dict__") if type(owner) is type else None
        current = declared.get(name) if declared is not None else None
        if current is not original:
            current = inspect.getattr_static(owner, name)
        if current is not original or not _definition_current(record):
            raise ConsoleProjectionUnavailable("Console reader definitions changed")
    if any(
        getattr(ConsoleProjectionView, name) is not record[0]
        or not _definition_current(record)
        for name, record in _VIEW_READERS
    ):
        raise ConsoleProjectionUnavailable("Console reader definitions changed")
