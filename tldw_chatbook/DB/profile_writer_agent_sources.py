"""Installed FULL transactions whose admission depends on a live local owner."""

from __future__ import annotations

import inspect
import sqlite3
import sys
from types import FunctionType


# Keep admission attached to the original transaction entry, never a request
# to execute client-supplied SQL. A planning or execution method may do work
# before or after its transaction without holding a grant during that work.
_SITES = {
    ("tldw_chatbook.DB.automatic_work", "AutomaticWorkLedger"): (
        "ledger",
        "reserve check_active admit_call commit prepare_chat_start accept_chat_start "
        "claim_wake _accept_attempt_transaction recover prepare_goal_iteration "
        "_prepare_workflow claim_workflow_task resume_workflow_phase "
        "quiesce_workflow_phase settle_workflow_phase bind_workflow_task_run settle",
    ),
    ("tldw_chatbook.DB.goal_runs", "GoalRunsStore"): (
        "goal_store",
        "create control settle_control set_provisioning remove_payloads checkpoint_failed defer_ready resolve_recovery",
    ),
    ("tldw_chatbook.DB.workflow_runs", "WorkflowRunsStore"): (
        "workflow_store",
        "create save_plan set_status publish_result remove_payloads save_decision open_change_window anchor_change_window",
    ),
    ("tldw_chatbook.Agents.goal_run_service", "GoalRunService"): (
        "goal_service",
        "resume review_result project_recovery",
    ),
    ("tldw_chatbook.Agents.workflow_run_service", "WorkflowRunService"): (
        "workflow_service",
        "checkpoint_task settle_execution_phase project_recovery resume review_quality "
        "record_returned_planning resolve_recovery",
    ),
    ("tldw_chatbook.Chat.console_goal_runs", "GoalIterationAuthorization"): (
        "goal_iteration", "record_outcome",
    ),
}
_PLANNING = "workflow_preparation.abort_planning"
AGENT_LOCAL_GRANT_KINDS = frozenset(
    f"{prefix}.{name.lstrip('_')}"
    for prefix, names in _SITES.values()
    for name in names.split()
) | {_PLANNING}
_methods = {}
_planning = None
_ledger_entries = ()


def register_agent_grant_sources(cls):
    """Capture the installed method, receiver class and actual transaction code."""
    global _ledger_entries
    prefix, names = _SITES[(cls.__module__, cls.__name__)]
    if prefix == "ledger":
        _ledger_entries = tuple(
            (name, vars(cls)[name], vars(cls)[name].__code__)
            for name in ("transaction", "_admission_transaction")
        )
    for name in names.split():
        descriptor = vars(cls)[name]
        original = inspect.unwrap(descriptor)
        _methods[original.__code__] = (
            f"{prefix}.{name.lstrip('_')}", cls, name, descriptor, original,
        )
    return cls


def register_planning_abort(function):
    """Capture the original prepared-planning return at module publication."""
    global _planning
    if (function.__module__, function.__name__) != (
        "tldw_chatbook.Chat.console_workflow_preparation", "_abort_workflow_planning"
    ):
        raise ValueError("Unexpected workflow abort source")
    _planning = function, function.__code__
    return function


def local_ledger_transaction(ledger, frame):
    """Derive the finite grant from the installed caller at context creation."""
    db = ledger._db
    if getattr(db, "_writer_profile", None) is None:
        return None
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from .automatic_work import AutomaticWorkLedger
    from .profile_writer_grant import _local_agent_transaction

    caller = sys._getframe(1)
    if (type(ledger) is not AutomaticWorkLedger or db.automatic_work is not ledger
        or caller.f_back is not frame or caller.f_locals.get("self") is not ledger
        or not any(
            caller.f_code is code and method.__code__ is code
            and vars(AutomaticWorkLedger).get(name) is method and name not in vars(ledger)
            for name, method, code in _ledger_entries
        )):
        raise RecoveryRequired("profile_writer_grant_source_changed")
    if _planning is not None and frame.f_code is _planning[1]:
        original, code = _planning
        if (original.__code__ is not code or frame.f_globals is not original.__globals__
            or frame.f_globals.get(original.__name__) is not original
            or frame.f_locals.get("ledger") is not ledger):
            raise RecoveryRequired("profile_writer_grant_source_changed")
        kind = _PLANNING
    else:
        site = _methods.get(frame.f_code)
        if site is None:
            return None
        kind, cls, name, descriptor, original = site
        owner = frame.f_locals.get("self")
        if (type(owner) is not cls or name in vars(owner) or vars(cls).get(name) is not descriptor
            or type(original) is not FunctionType or original.__code__ is not frame.f_code
            or frame.f_globals is not original.__globals__):
            raise RecoveryRequired("profile_writer_grant_source_changed")
        if kind == "goal_iteration.record_outcome":
            if owner.context.ledger is not ledger or owner._coordinator.service.db is not db:
                raise RecoveryRequired("profile_writer_grant_source_changed")
        elif owner is not ledger and getattr(owner, "db", None) is not db:
            raise RecoveryRequired("profile_writer_grant_source_changed")
    connection = getattr(db._thread_local, "conn", None)
    if connection is not None:
        try:
            if sqlite3.Connection.in_transaction.__get__(connection):
                return None  # The original FULL transaction rejects nesting.
        except sqlite3.ProgrammingError:
            pass
    return _local_agent_transaction(db, kind)
