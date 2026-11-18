"""Named AgentRuns commands for original, data-only work and accounting transactions."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import wraps
from types import FunctionType, MappingProxyType


_STORES = {}
_COMMANDS = {}


@contextmanager
def granted_connection(database, grant_scope, *, execution=None):
    """Retire the bound native handle after its original operation scopes exit."""
    with grant_scope as grant:
        bound = None
        try:
            with ExitStack() as original:
                if execution is not None:
                    original.enter_context(execution)
                connection = original.enter_context(database.connection())
                grant.bind(connection)
                bound = connection
                yield connection, grant
        finally:
            if bound is not None:
                grant.retire(bound)


def _mapping(value):
    if type(value) is MappingProxyType:
        return dict(value)
    raise TypeError("Unsupported agent-work value")


def _snapshot_maps(value, factory):
    if not hasattr(value, "accounting"):
        return value
    from tldw_chatbook.Agents.goal_models import GoalSnapshot
    from tldw_chatbook.Agents.workflow_models import WorkflowSnapshot

    if type(value) not in (GoalSnapshot, WorkflowSnapshot):
        return value
    accounting = value.accounting
    accounting = replace(accounting, **{
        field: factory(dict(getattr(accounting, field)))
        for field in ("used", "reserved", "available")
    })
    return value.model_copy(update={"accounting": accounting})


def _original_store(kind, store):
    cls, definitions = _STORES[kind]
    return type(store) is cls and all(
        name not in vars(store) and vars(cls).get(name) is descriptor and function.__code__ is code
        for name, descriptor, function, code in definitions
    )


def _client_database(kind, owner):
    from .AgentRuns_DB import AgentRunsDB

    db = owner._db if kind in ("ledger", "worktree") else owner.db
    if type(db) is not AgentRunsDB or db._writer_profile is None or not _original_store(kind, owner):
        return None
    if kind == "goal" and owner.payload_limit != owner._DEFAULT_PAYLOAD_LIMIT:
        return None
    conn = getattr(db._thread_local, "conn", None)
    if conn is not None:
        try:
            if sqlite3.Connection.in_transaction.__get__(conn):
                return None  # Preserve the original transaction/nesting refusal.
        except sqlite3.ProgrammingError:
            pass
    if kind != "worktree":
        ledger = db.automatic_work
        if (not _original_store("ledger", ledger) or kind == "ledger" and owner is not ledger
            or ledger._wall_clock is not time.time or ledger._monotonic_clock is not time.monotonic):
            # A live accounting capability, clock, or validator belongs to the
            # executor. Its original FULL transaction retains those checks.
            return None
    return db


def _routed(kind, name, method):
    operation = f"agent_runs.work.{kind}.{name.lstrip('_')}"
    _COMMANDS[operation] = kind, method
    if sys.platform != "win32":
        return method

    @wraps(method)
    def routed(owner, *args, **kwargs):
        db = _client_database(kind, owner)
        if db is None:
            return method(owner, *args, **kwargs)
        from tldw_chatbook.Backup_Recovery.participants import core_access
        from .profile_writer import ProfileWriterOutcomeUnknown, call_bound_profile_writer
        from .profile_writer_commands import _run_contract

        core_access(db)
        signature, request, result_type = _run_contract(method)
        values = dict(signature.bind(owner, *args, **kwargs).arguments)
        del values["self"]
        exact_model = {
            "create": "request", "set_provisioning": "snapshot",
            "resolve_recovery": "resolution", "save_plan": "plan",
        }.get(name)
        if exact_model is not None and type(values[exact_model]) is not request.model_fields[exact_model].annotation:
            # Run the original exact-class refusal. JSON would erase a subtype.
            return method(owner, *args, **kwargs)
        if kind == "workflow" and name == "baseline_change_window":
            values["roots"] = tuple(str(root) for root in values["roots"])
        if kind == "workflow" and name == "publish_change_window":
            values["records"] = tuple(values["records"])
        if kind == "ledger" and name == "create_chain" and values.get("limits") is None:
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

            values["limits"] = AutomaticWorkLimits.from_settings()
        if kind == "goal" and name == "set_provisioning":
            values["snapshot"] = _snapshot_maps(values["snapshot"], dict)
        if (
            kind == "goal" and name != "clear_wait" or kind == "workflow" and name in {
                "create", "save_plan", "set_status", "publish_result", "remove_payloads", "save_decision"
            } or kind == "ledger" and name == "settle"
        ) and db.automatic_work._workflow_accounting_validators is not None:
            # A snapshot with live clock evidence may invoke the originating
            # runtime's registered proof. Retain that transaction on its owner.
            return method(owner, *args, **kwargs)
        encoded = request.model_validate(values, strict=True).model_dump(mode="json", fallback=_mapping)
        data = call_bound_profile_writer(db, db._writer_profile, operation, encoded)
        try:
            result = result_type.validate_json(json.dumps(data, allow_nan=False), strict=True)
            return _snapshot_maps(result, MappingProxyType)
        except (TypeError, ValueError) as error:
            raise ProfileWriterOutcomeUnknown("Invalid agent-work mutation receipt") from error

    return routed


def writer_work_store(kind, *methods):
    """Register only the listed complete transactions of this concrete stock store."""
    def register(cls):
        for name in methods:
            setattr(cls, name, _routed(kind, name, vars(cls)[name]))
        records = []
        for name, descriptor in vars(cls).items():
            function = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
            if isinstance(function, FunctionType):
                records.append((name, descriptor, function, function.__code__))
        _STORES[kind] = cls, tuple(records)
        return cls

    return register


def execute_agent_work_command(profile, operation, arguments):
    """Validate before opening the writer database, then run the original FULL/CAS operation."""
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from . import agent_worktrees, automatic_work, goal_runs, workflow_runs  # noqa: F401 - register stock methods
    from .profile_writer import ProfileWriterOutcomeUnknown, executing_writer_repository
    from .profile_writer_commands import _run_contract, agent_runs_command_database

    command = _COMMANDS.get(operation)
    if command is None or type(arguments) is not dict:
        raise ValueError("Unknown agent-work writer command")
    kind, method = command
    _, request, result_type = _run_contract(method)
    try:
        values = vars(request.model_validate_json(json.dumps(arguments, allow_nan=False)))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid agent-work writer arguments") from error
    if not executing_writer_repository(profile, "db.agent_runs"):
        raise RecoveryRequired("agent_runs_writer_selection_changed")
    with agent_runs_command_database(profile) as db:
        if kind == "worktree":
            owner = agent_worktrees.AgentWorktreeRepository(db)
        elif kind == "goal":
            owner = db.goal_runs
        elif kind == "workflow":
            owner = db.workflow_runs
        else:
            owner = db.automatic_work
        result = method(owner, **values)
        try:
            result = result_type.validate_python(result, strict=True)
            return result_type.dump_python(_snapshot_maps(result, dict), mode="json", fallback=_mapping)
        except (TypeError, ValueError) as error:
            raise ProfileWriterOutcomeUnknown("Agent-work mutation receipt unavailable") from error
