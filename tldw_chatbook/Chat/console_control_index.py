"""Finite index mutations on the original service, run and native owners."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager


def bind_index_owner(controls, owner):
    if owner is None:
        return
    if owner._console_controls not in (None, controls):
        raise RuntimeError("Console index control owner changed")
    if owner.service is not None:
        bind_index_service(controls, owner, owner.service)
    owner._console_controls = controls


def bind_index_service(controls, owner, service):
    with service._lock:
        previous = getattr(service, "_console_control_owner", None)
        if previous is not None:
            if previous is not owner:
                raise RuntimeError("Console index service owner changed")
            return
        if (
            service._operations
            or service._searches
            or any(not run.native_retired for run in service._runs.values())
            or service._cleanup_thread is not None
            and service._cleanup_thread.is_alive()
        ):
            raise RuntimeError("Console index has unrecorded active work")
        service._console_control_owner = owner


def index_source(service):
    owner = getattr(service, "_console_control_owner", None)
    if owner is None:
        return None
    from .console_control_dispatch import require_control_source

    controls = owner._console_controls
    if (
        controls is None
        or controls._domain._project_index_owner is not owner
        or owner.service is not service
    ):
        raise RuntimeError("Console index receiver changed")
    retained = getattr(service._operation_local, "control_source", None)
    source = (
        retained if retained is not None else require_control_source(controls._runtime)
    )
    if source.controls is not controls:
        raise RuntimeError("Console index command source changed")
    return source


def begin_index_operation(service, operation):
    source = index_source(service)
    if source is None:
        return None
    return source.controls, source.controls._begin_effect(
        "index_operation", operation, source
    )


def finish_index_operation(effect, operation):
    if effect is not None:
        controls, token = effect
        controls._finish_effect(
            token,
            retired=(
                operation.done.is_set()
                and not operation.retirement_unproven
                and not operation.retained
            ),
        )


@contextmanager
def index_composition(owner):
    controls = owner._console_controls
    if controls is None:
        yield
        return
    from .console_control_dispatch import require_control_source

    source = require_control_source(controls._runtime)
    effect = controls._begin_effect("index_composition", owner, source)
    retired = False
    unpublished_count = len(owner._unpublished)
    try:
        yield
    finally:
        try:
            candidate = owner.service
            if candidate is None and len(owner._unpublished) == unpublished_count + 1:
                candidate = owner._unpublished[-1]
            if candidate is not None:
                candidate._factory.assert_retired()
                retired = candidate._factory._composition_pin.closed
        finally:
            controls._finish_effect(effect, retired=retired)


@contextmanager
def index_writer_stage(service, run, store, kind):
    """End only this transaction; the original run keeps its pin and handles."""
    source = run.control_source
    if source is None:
        yield
        return
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import (
        core_operation,
        repository_participant,
        core_cached_connection,
    )
    from tldw_chatbook.MCP.local_permission_operation import _hold_retired_or_shared

    controls = source.controls
    owner = getattr(service, "_console_control_owner", None)
    thread = threading.current_thread()
    if (
        kind not in {"admit", "emit", "final"}
        or owner is None
        or owner._console_controls is not controls
        or owner.service is not service
        or service._runs.get(run.claim.job_id) is not run
        or run.control_store is not store
        or run.native_retired
        or thread is not (run.control_retire_thread if kind == "final" else run.writer)
        or run.pin is None
        or run.pin.closed
        or run.lease is None
        or run.lease.closed
    ):
        raise RuntimeError("Console index stage owner changed")
    effect = controls._begin_effect("index_" + kind, run, source)
    operation = None
    retired = False
    repositories = (store, owner.registry.db)
    previous = tuple(
        core_cached_connection(
            repository, getattr(repository._thread_local, "conn", None)
        )
        for repository in repositories
    )
    try:
        with core_operation(store):
            operation = storage._operation_local.operation
            yield
    finally:
        try:
            if kind == "final":
                # After the counted transaction exits, use the original exact
                # native close before publishing this final stage's retirement.
                store.close_native()
            for repository, borrowed in zip(repositories, previous, strict=True):
                connection = core_cached_connection(
                    repository, getattr(repository._thread_local, "conn", None)
                )
                if connection is not None:
                    if sqlite3.Connection.in_transaction.__get__(connection):
                        raise RuntimeError("Console index transaction did not settle")
                    with storage._lock:
                        lease = repository_participant(repository).connections.get(
                            connection
                        )
                        if (
                            lease is None
                            or lease not in storage._live_leases
                            or lease.resource_thread is not thread
                        ):
                            raise RuntimeError("Console index connection owner changed")
                    # New held handles stay with the exact original run. Their
                    # original writer/registry/supervisor closes remain required.
                    if not any(row[2] is connection for row in run.control_connections):
                        run.control_connections.append(
                            (thread, repository, connection, lease, borrowed)
                        )
            with storage._lock:
                retired = (
                    operation is not None
                    and operation not in storage._operations
                    and operation.lease not in storage._live_leases
                    and operation.lease._key is None
                    and _hold_retired_or_shared(operation.hold, storage)
                )
        finally:
            controls._finish_effect(effect, retired=retired)


@contextmanager
def index_thread_source(service, source):
    previous = getattr(service._operation_local, "control_source", None)
    service._operation_local.control_source = source
    try:
        yield
    finally:
        service._operation_local.control_source = previous


def require_run_native_retirement(run):
    """Verify the original run's recorded handles after its actual closes."""
    if run.control_source is None:
        return
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import repository_participant

    for _thread, repository, connection, lease, _borrowed in run.control_connections:
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            pass
        else:
            raise RuntimeError("Console index native connection remains open")
        with storage._lock:
            if (
                connection in repository_participant(repository).connections
                or lease in storage._live_leases
                or lease._key is not None
            ):
                raise RuntimeError("Console index native lease remains owned")
