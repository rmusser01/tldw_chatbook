"""One source-bound observation of original persisted Agent run records."""

from __future__ import annotations

import asyncio
import json
from types import MethodType

from ..DB import base_db
from ..DB.AgentRuns_DB import AgentRunsDB
from ..Utils.definition_records import definitions_current
from . import console_change_review_read as marker_reads
from . import console_preparation_reads as preparation_reads
from .console_agent_bridge import ConsoleAgentBridge
from .console_domain_services import _definition, _definition_current
from .console_execution_protocol import (
    ConsoleProtocolError,
    _array,
    _boolean,
    _integer,
    _detach,
    _optional,
    _record,
    _text,
    _unique_pairs,
)


_TEXT = _text()
_REQUEST = _record(
    conversation_id=_text(minimum=1),
    include_superseded=_boolean,
    limit=_optional(_integer()),
    agent_kind=_optional(_TEXT),
)
_FIELDS = (
    "id",
    "conversation_id",
    "parent_run_id",
    "agent_kind",
    "status",
    "created_at",
    "updated_at",
    "assistant_message_id",
)
_ROW = _record(
    **{
        name: _optional(_TEXT)
        if name in {"parent_run_id", "assistant_message_id"}
        else _TEXT
        for name in _FIELDS
    },
    steps=_TEXT,
)


def validate_run_history(value):
    frozen = _detach(value, "run_history", 0, set(), thaw=False)
    _array(_ROW)(frozen, "run_history")
    for row in value:
        try:
            steps = json.loads(row["steps"], object_pairs_hook=_unique_pairs)
        except (ValueError, RecursionError):
            raise ConsoleProtocolError("invalid_run_history_steps") from None
        if type(steps) is not list or any(type(item) is not dict for item in steps):
            raise ConsoleProtocolError("invalid_run_history_steps")
        _detach(steps, "run_history.steps", 0, set(), thaw=False)


def decode_run_history(value):
    validate_run_history(value)
    return [{**row, "steps": json.loads(row["steps"])} for row in value]


class ConsoleBackendRunHistoryRead:
    def __init__(self, presentation, identity, request):
        from .console_control_dispatch import require_control_source

        self.presentation, self.identity = presentation, identity
        self.controls = presentation.backend.controls
        self.source = require_control_source(presentation._runtime)
        self.request = _detach(request, "run_history", 0, set(), thaw=False)
        _REQUEST(self.request, "run_history")
        self.bridge = presentation._runtime._agent_bridge
        self.database = self.bridge._db if self.bridge is not None else None
        self.local = getattr(self.database, "_thread_local", None)
        self.path = getattr(self.database, "db_path", None)
        self.task = asyncio.current_task()
        self.reads = set()
        self.observation = None
        self.require_current()

    def require_current(self):
        self.presentation.require_association(self.presentation.association)
        backend = self.presentation.backend
        backend.require_sources()
        session = backend.store._sessions.get(self.identity.session_id)
        if (
            self.controls is not backend.controls
            or self.source.controls is not self.controls
            or self.source.identity is not self.identity
            or session is None
            or session.incarnation_id != self.identity.incarnation_id
            or session.persisted_conversation_id != self.request["conversation_id"]
            or backend.runtime._agent_bridge is not self.bridge
            or type(self.bridge) is not ConsoleAgentBridge
            or self.bridge._db is not self.database
            or type(self.database) is not AgentRunsDB
            or self.database._thread_local is not self.local
            or self.database.db_path != self.path
            or self.database.is_memory_db
            or marker_reads._factory_sources_current is not _FACTORY[0]
            or not _definition_current(_FACTORY)
            or not _FACTORY[0](self.database)
            or any(
                getattr(type(self), name) is not record[0]
                or not _definition_current(record)
                for name, record in _METHODS
            )
            or not definitions_current(_CONNECTIONS)
            or preparation_reads.run_preparation_read is not _PREPARATION[0]
            or not _definition_current(_PREPARATION)
        ):
            raise RuntimeError("Console run history source changed")
        for name, record in _DATABASE:
            method = getattr(self.database, name)
            if (
                type(method) is not MethodType
                or method.__self__ is not self.database
                or method.__func__ is not record[0]
                or not _definition_current(record)
            ):
                raise RuntimeError("Console original history reader changed")

    def read(self):
        self.require_current()
        with base_db.operation_owned_connection(self.database):
            rows = self.database.list_runs(**self.request)
        self.require_current()
        # These are original persisted trace facts. Routing/continuation state,
        # model configuration, result bodies and definition snapshots stay local.
        values = [
            {
                **{name: row[name] for name in _FIELDS},
                "steps": json.dumps(row["steps"], ensure_ascii=False, allow_nan=False),
            }
            for row in rows
        ]
        validate_run_history(values)
        return tuple(values)

    async def run(self):
        self.require_current()
        result = await _PREPARATION[0](
            self.read,
            creator=self,
            session_id=self.identity.session_id,
            reads=self.reads,
            observers=(self.presentation._runtime._preparation_reads,),
            require_current=self.require_current,
            source=self.source,
        )
        if self.observation is None:
            raise RuntimeError("Console original history read was not enrolled")
        return result

    def register(self, read, source):
        from .console_execution_retirement import _ReadResources

        self.require_current()
        callback = read.callback
        if (
            source is not self.source
            or read.task is not self.task
            or read.task is not asyncio.current_task()
            or read.creator is not self
            or read not in self.reads
            or self.observation is not None
            or self.controls._settings_command_tasks.get(self.task) is not source
            or type(callback) is not MethodType
            or callback.__self__ is not self
            or callback.__func__ is not dict(_METHODS)["read"][0]
        ):
            raise RuntimeError("Console original history handoff changed")
        observation = self.observation = _ReadResources(source, read)
        observation.effect = self.controls._begin_effect(
            "run_history_read", observation, source
        )
        return observation


_METHODS = tuple(
    (name, _definition(getattr(ConsoleBackendRunHistoryRead, name)))
    for name in ("__init__", "require_current", "read", "run", "register")
)
_DATABASE = tuple(
    (name, _definition(getattr(AgentRunsDB, name)))
    for name in ("list_runs", "_rows_to_dicts", "_batch_hydrate_steps")
)
_CONNECTIONS = base_db._KEY_PROCESS_DEFINITIONS
_FACTORY = _definition(marker_reads._factory_sources_current)
_PREPARATION = _definition(preparation_reads.run_preparation_read)
