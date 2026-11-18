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
        "create control settle_control set_provisioning remove_payloads",
    ),
    ("tldw_chatbook.DB.workflow_runs", "WorkflowRunsStore"): (
        "workflow_store",
        "create save_plan set_status publish_result remove_payloads save_decision open_change_window",
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
}
_PLANNING = "workflow_preparation.abort_planning"
AGENT_LOCAL_GRANT_KINDS = frozenset(
    f"{prefix}.{name.lstrip('_')}"
    for prefix, names in _SITES.values()
    for name in names.split()
) | {_PLANNING}
_methods = {}
_planning = None


def register_agent_grant_sources(cls):
    """Capture the installed method, receiver class and actual transaction code."""
    prefix, names = _SITES[(cls.__module__, cls.__name__)]
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
        or caller.f_code not in (AutomaticWorkLedger.transaction.__code__,
                                 AutomaticWorkLedger._admission_transaction.__code__)):
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
            or frame.f_globals is not original.__globals__
            or (owner is not ledger and getattr(owner, "db", None) is not db)):
            raise RecoveryRequired("profile_writer_grant_source_changed")
    connection = getattr(db._thread_local, "conn", None)
    if connection is not None:
        try:
            if sqlite3.Connection.in_transaction.__get__(connection):
                return None  # The original FULL transaction rejects nesting.
        except sqlite3.ProgrammingError:
            pass
    return _local_agent_transaction(db, kind)
