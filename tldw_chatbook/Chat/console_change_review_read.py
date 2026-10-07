"""Finite worker ownership for stock Change Review transcript reads."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ..Backup_Recovery.participants import (
    _core_cached_connection,
    _core_closing,
    _core_operation,
)
from ..DB.AgentRuns_DB import AgentRunsDB, _CHANGE_REVIEW_READ_SOURCES
from . import console_agent_bridge as bridge_module
from .console_preparation_reads import run_preparation_read

_BRIDGE_SOURCES = bridge_module._CHANGE_REVIEW_MARKER_SOURCES
_HELPER_SOURCE = bridge_module._CHANGE_REVIEW_MARKER_HELPER_SOURCE


class MarkerReadObsolete(RuntimeError):
    """A display read lost its captured source; its result cannot be published."""


def _function_current(record):
    function, code, defaults, kwdefaults, items = record
    return (
        function.__code__ is code
        and function.__defaults__ is defaults
        and function.__kwdefaults__ is kwdefaults
        and tuple((function.__kwdefaults__ or {}).items()) == items
    )


def _sources_current(receiver, sources):
    for name, descriptor, records in sources:
        if inspect.getattr_static(receiver, name, None) is not descriptor:
            return False
        if not all(_function_current(record) for record in records):
            return False
    return True


@dataclass
class _MarkerRead:
    runtime: Any
    coordinator: Any
    bridge: Any
    database: AgentRunsDB
    conversation_id: str
    local: Any
    path: Any
    reader: Any
    renderer: Any
    active_connection: Any = None

    def require_current(self):
        if (
            self.runtime._disposed
            or self.runtime._agent_bridge is not self.bridge
            or self.runtime._change_review_coordinator is not self.coordinator
            or self.bridge._db is not self.database
            or self.database._thread_local is not self.local
            or self.database.db_path != self.path
            or self.database.is_memory_db
            or (
                self.active_connection is not None
                and getattr(self.local, "conn", None) is not self.active_connection
            )
            or not _sources_current(self.database, _CHANGE_REVIEW_READ_SOURCES)
            or not _sources_current(self.bridge, _BRIDGE_SOURCES)
            or bridge_module._CHANGE_REVIEW_MARKER_SOURCES is not _BRIDGE_SOURCES
            or not _function_current(_HELPER_SOURCE)
            or bridge_module._read_change_review_markers is not self.reader
        ):
            raise MarkerReadObsolete("change_review_marker_source_changed")

    def read(self):
        self.require_current()
        previous = _core_cached_connection(
            self.database, getattr(self.local, "conn", None)
        )
        connection = None
        try:
            with _core_operation(self.database):
                self.require_current()
                connection = self.database._held_connection()
                self.active_connection = connection
                self.require_current()
                if getattr(self.local, "conn", None) is not connection:
                    raise MarkerReadObsolete("change_review_marker_connection_changed")
                return self.reader(
                    self.database,
                    self.conversation_id,
                    self.renderer,
                    require_current=self.require_current,
                )
        finally:
            # The counted operation exits first. Retire the actual acquired A,
            # even if a callback installed a different cache B in the meantime.
            if connection is not None and connection is not previous:
                with _core_closing(self.database, connection) as allowed:
                    if not allowed:
                        raise RuntimeError(
                            "change_review_marker_connection_not_retired"
                        )
                    connection.close()
                    if getattr(self.local, "conn", None) is connection:
                        self.local.conn = None
            self.active_connection = None

    async def run(self, projection):
        return await run_preparation_read(
            self.read,
            creator=projection,
            session_id=None,
            reads=projection._preparation_reads,
            observers=(self.runtime._preparation_reads,),
            require_current=self.require_current,
        )


def capture_marker_read(runtime, coordinator, bridge, conversation_id):
    """Select only original file-backed readers; custom callers keep their ABI."""
    from .console_runtime import ConsoleRuntime

    if (
        type(runtime) is not ConsoleRuntime
        or type(bridge) is not bridge_module.ConsoleAgentBridge
    ):
        return None
    database = bridge._db
    if (
        type(database) is not AgentRunsDB
        or database.is_memory_db
        or conversation_id is None
        or coordinator is None
        or not _sources_current(database, _CHANGE_REVIEW_READ_SOURCES)
        or not _sources_current(bridge, _BRIDGE_SOURCES)
        or bridge_module._CHANGE_REVIEW_MARKER_SOURCES is not _BRIDGE_SOURCES
        or bridge_module._read_change_review_markers is not _HELPER_SOURCE[0]
        or not _function_current(_HELPER_SOURCE)
    ):
        return None
    return _MarkerRead(
        runtime,
        coordinator,
        bridge,
        database,
        conversation_id,
        database._thread_local,
        database.db_path,
        bridge_module._read_change_review_markers,
        bridge._change_review_marker_block,
    )
