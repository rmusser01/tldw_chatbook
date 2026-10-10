"""Bounded exact core admission diagnostic; never replaces or drains a producer."""

import sys
import threading
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver


class OriginalCoreProducerDiagnostic(OriginalStorageUnitObserver):
    """Keep only active original admitted scopes for declared fixture creators."""

    def __init__(self, creators):
        super().__init__({}, lambda: False, lambda _name: None)
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        self.storage = storage
        self.participants = tuple(row[3] for row in creators.owners)
        self.operation_factory = storage._repository_operation
        self.generator = self.operation_factory.__wrapped__
        self.code = self._pin(self.generator)
        self.work_code = self._pin(_WorkItem.run)
        self.slots.extend(
            (
                (storage, "_repository_operation", self.operation_factory),
                (self.operation_factory, "__wrapped__", self.generator),
                (_WorkItem, "run", _WorkItem.run),
            )
        )
        self.scopes = {}
        self.invalid_overflow = 0

    def install(self):
        for tool in range(5, 0, -1):
            if tool == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(tool, "original-stock-core-producer")
            except ValueError:
                continue
            self.tool = tool
            break
        assert self.tool is not None
        self.active = self.installed = True
        try:
            for event, callback in (
                (sys.monitoring.events.PY_YIELD, self._yield),
                (sys.monitoring.events.PY_RETURN, self._return),
            ):
                assert (
                    sys.monitoring.register_callback(self.tool, event, callback) is None
                )
                self.registered[event] = callback
            self.codes = {self.code: "original_repository_admission"}
            assert sys.monitoring.get_local_events(self.tool, self.code) == 0
            sys.monitoring.set_local_events(
                self.tool,
                self.code,
                sys.monitoring.events.PY_YIELD | sys.monitoring.events.PY_RETURN,
            )
        except BaseException:
            self.close()
            raise

    def _yield(self, code, _offset, value):
        frame = parent = item = None
        try:
            if code is not self.code or type(value) is not self.storage._Operation:
                return
            if value.participant not in self.participants:
                return
            frame = sys._getframe(1)
            assert frame.f_code is code and frame.f_globals is vars(self.storage)
            if frame.f_locals.get("reuse") is True:
                return
            assert frame.f_locals.get("reuse") is False
            assert frame.f_locals.get("operation") is value
            assert (
                value in self.storage._operations
                and value.thread is threading.current_thread()
            )
            if value.thread is threading.main_thread():
                return
            lead = []
            parent = frame.f_back
            for _ in range(24):
                if parent is None or parent.f_code is self.work_code:
                    break
                if len(lead) < 16:
                    lead.append(
                        (
                            parent.f_globals.get("__name__"),
                            parent.f_code.co_qualname,
                            parent.f_code.co_firstlineno,
                        )
                    )
                parent = parent.f_back
            if parent is None or parent.f_code is not self.work_code:
                return
            item = parent.f_locals.get("self")
            assert type(item) is _WorkItem and type(item.future) is Future
            assert item.future.running() and not item.future.done()
            assert id(frame) not in self.scopes and len(self.scopes) < 16
            self.scopes[id(frame)] = (
                value,
                threading.current_thread(),
                item.future,
                tuple(lead),
            )
        except BaseException as error:
            if len(self.invalid) < 16:
                self.invalid.append(type(error).__name__ + ":original_core_scope")
            else:
                self.invalid_overflow += 1
        finally:
            del frame, parent, item, value

    def _return(self, code, _offset, value):
        frame = None
        try:
            if code is not self.code:
                return
            frame = sys._getframe(1)
            assert frame.f_code is code and frame.f_globals is vars(self.storage)
            row = self.scopes.pop(id(frame), None)
            if row is not None:
                assert threading.current_thread() is row[1]
                assert row[0] not in self.storage._operations
        except BaseException as error:
            if len(self.invalid) < 16:
                self.invalid.append(type(error).__name__ + ":original_core_return")
            else:
                self.invalid_overflow += 1
        finally:
            del frame, value

    def snapshot(self):
        """Emit only admitted actors and unqualified primitive caller-code leads."""
        with self.storage._lock:
            return [
                {
                    "owner": operation.participant.owner_id,
                    "thread": thread.name,
                    "exact_original_Future_running": future.running(),
                    "exact_original_Future_done": future.done(),
                    "operation_still_counted": operation in self.storage._operations,
                    "caller_code_lead_only": lead,
                    "diagnostic_only_no_native_producer_retirement_claim": True,
                }
                for operation, thread, future, lead in tuple(self.scopes.values())
            ]

    def close(self):
        receipt = super().close()
        receipt["explicit_active_operation_Future_custody"] = bool(self.scopes)
        receipt["frames_retained"] = False
        receipt["diagnostic_only"] = True
        receipt["invalid_overflow"] = self.invalid_overflow
        return receipt
