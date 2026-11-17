"""Exact native observations for one selected, synchronous local operation.

Original raw/core owners retain their resources. This context only suppresses
nested permission transport and observes their original retirement boundaries.
"""

from __future__ import annotations

import os
import threading

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

_local = threading.local()


def active(source=None):
    operation = getattr(_local, "operation", None)
    if operation is None:
        return None
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    if type(operation) is not LocalPermissionOperation or operation.creator != (
        os.getpid(),
        threading.current_thread(),
        storage._task_identity(),
    ):
        raise RecoveryRequired("permission_operation_creator_changed")
    if source is not None and source is not operation.source:
        return None
    return operation


def selected(source):
    operation = active(source)
    if operation is None:
        return False
    operation.require_current()
    return True


def enroll(kind, value):
    operation = active()
    if operation is None:
        return
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    if kind == "lease":
        value = (value, storage._holds.get(value._key))
    operation.observations[kind].append(value)


def _hold_retired_or_shared(hold, storage):
    if hold is None:
        return True
    if hold in storage._holds.values():
        return (
            hold.error is None
            and not hold.stop.is_set()
            and hold.ready.is_set()
            and hold.thread.is_alive()
            and hold.count > 0
            and hold.count
            == sum(lease._key == hold.key for lease in storage._live_leases)
        )
    return (
        not hold.thread.is_alive()
        and hold not in storage._retiring_holds
        and hold.count == 0
        and (hold.error is None or isinstance(hold.error, storage.AdmissionCancelled))
    )


def _state_resources_retired(state, storage):
    return (
        not state.active
        and not state.uncertain
        and not state.files
        and not state.descriptors
        and not state.pins
        and state.companion_guard is None
        and all(
            lease not in storage._live_leases and lease._key is None
            for lease in state.leases
        )
        and all(_hold_retired_or_shared(hold, storage) for hold in state.holds)
    )


def before_complete(state):
    """Check this exact retired raw owner before its original stamps publish."""
    operation = active(state.source)
    if operation is None:
        return
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    operation.require_current()
    with storage._lock:
        if not _state_resources_retired(state, storage):
            operation.refuse_retirement()


class LocalPermissionOperation:
    """Observe one original complete action on its native creator thread."""

    def __init__(self, source, require_current):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        self.source = source
        self.require_current = require_current
        self.creator = (
            os.getpid(),
            threading.current_thread(),
            storage._task_identity(),
        )
        self.observations = {
            name: [] for name in ("acquisition", "lease", "raw", "descriptor", "file")
        }
        self.retired = False
        self.retirement_error = None

    def __enter__(self):
        if active() is not None:
            raise RecoveryRequired("nested_local_permission_operation")
        self.require_current()
        _local.operation = self
        return self

    def __exit__(self, _exc_type, error, _traceback):
        try:
            try:
                self.verify_retirement()
            except RecoveryRequired as retirement_error:
                self.retirement_error = retirement_error
                # Original lifecycle reconciliation and native failures keep
                # their exact outcome. The original retained owners separately
                # remain unresolved; wrapper return is not native retirement.
                if error is None:
                    raise
        finally:
            _local.operation = None

    def refuse_retirement(self):
        self.source._permission_publication.fail()
        self.source._mcp_persistence_error = "mcp_persistence_incomplete"
        raise RecoveryRequired("permission_native_retirement_unproven")

    def verify_retirement(self):
        """Validate exact enrollment, permitting unrelated and shared owners."""
        from tldw_chatbook.Backup_Recovery import raw_participants as raw
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        with storage._lock:
            valid = all(
                item not in storage._pending_acquisitions
                for item in self.observations["acquisition"]
            )
            valid &= all(
                handle.closed and native.closed
                for handle, native in self.observations["file"]
            )
            for operation, state in self.observations["raw"]:
                valid &= (
                    operation not in raw._states
                    and operation not in storage._raw_operations
                    and _state_resources_retired(state, storage)
                )
            for lease, hold in self.observations["lease"]:
                valid &= lease not in storage._live_leases and lease._key is None
                valid &= _hold_retired_or_shared(hold, storage)
            if not valid:
                self.refuse_retirement()
            self.retired = True
