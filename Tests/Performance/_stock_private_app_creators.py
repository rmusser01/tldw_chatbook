"""Test-only native prerequisite; compile/review before installation."""

import inspect
import sqlite3
import sys
import threading
from pathlib import Path
from types import FunctionType
from types import MethodType
from concurrent.futures import Future
from concurrent.futures.thread import _WorkItem

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver


class OriginalPrivateAppCreators(OriginalStorageUnitObserver):
    """Capture only original private-profile births under the declared App call."""

    def __init__(self, app_source, root):
        super().__init__({}, lambda: False, lambda _name: None)
        from tldw_chatbook.Backup_Recovery import participants
        from tldw_chatbook.DB.Evals_DB import EvalsDB
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

        self.root = root.resolve()
        self.creator = threading.current_thread()
        self.app_source = app_source
        self.app_type = app_source.TldwCli
        app_init = inspect.getattr_static(self.app_type, "__init__")
        self.app_code = self._pin(app_init)
        self.slots.extend(
            (
                (app_source, "TldwCli", self.app_type),
                (self.app_type, "__init__", app_init),
            )
        )
        self.original_types = participants._repository_types()
        assert EvalsDB in self.original_types and SubscriptionsDB in self.original_types
        self.roles, self.births, self.owners = {}, {}, []
        self.parallel_methods = {}
        for name in ("_init_notes_service", "_init_prompts_service", "_init_media_db"):
            function = inspect.getattr_static(self.app_type, name)
            self.parallel_methods[self._pin(function)] = function
            self.slots.append((self.app_type, name, function))
        self.timed = inspect.getattr_static(self.app_type, "_timed_init_task")
        self.timed_code = self._pin(self.timed)
        self.work_code = self._pin(_WorkItem.run)
        self.slots.extend(
            (
                (self.app_type, "_timed_init_task", self.timed),
                (_WorkItem, "run", _WorkItem.run),
            )
        )
        self.parallel_producers = []
        self.app_frame = None
        self.app_returned = False
        for owner in self.original_types:
            init = inspect.getattr_static(owner, "__init__")
            assert type(init) is FunctionType
            self.roles[self._pin(init)] = owner
            self.slots.append((owner, "__init__", init))
        self.target = None
        self.retired = False

    def install(self):
        for tool in range(5, 0, -1):
            if tool == sys.monitoring.DEBUGGER_ID:
                continue
            try:
                sys.monitoring.use_tool_id(tool, "stock-private-App-creators")
            except ValueError:
                continue
            self.tool = tool
            break
        assert self.tool is not None
        self.active = self.installed = True
        try:
            for event, callback in (
                (sys.monitoring.events.PY_START, self._start),
                (sys.monitoring.events.PY_RETURN, self._return),
            ):
                assert (
                    sys.monitoring.register_callback(self.tool, event, callback) is None
                )
                self.registered[event] = callback
            self.codes = {code: "private_App_constructor" for code in self.roles}
            self.codes[self.app_code] = "declared_original_App_constructor"
            for code in self.codes:
                assert sys.monitoring.get_local_events(self.tool, code) == 0
                sys.monitoring.set_local_events(
                    self.tool,
                    code,
                    sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN,
                )
        except BaseException:
            self.close()
            raise

    def _start(self, code, _offset):
        frame = parent = owner = item = None
        try:
            frame = sys._getframe(1)
            assert frame.f_code is code
            owner = frame.f_locals.get("self")
            if code is self.app_code:
                assert threading.current_thread() is self.creator
                assert type(owner) is self.app_type and self.target is None
                assert frame.f_globals is vars(self.app_source)
                self.target, self.app_frame = owner, id(frame)
                return
            expected = self.roles[code]
            if type(owner) is not expected:
                return
            parent = frame.f_back
            for _ in range(24):
                if (
                    parent is None
                    or parent.f_code is self.app_code
                    or parent.f_code in self.parallel_methods
                ):
                    break
                parent = parent.f_back
            if parent is None:
                return
            if parent.f_globals is not vars(self.app_source):
                raise AssertionError("private_creator_App_source_changed")
            target = parent.f_locals.get("self")
            assert type(target) is self.app_type
            assert self.target is target
            if parent.f_code is self.app_code:
                assert threading.current_thread() is self.creator
                assert id(parent) == self.app_frame
            elif parent.f_code in self.parallel_methods:
                callback = self.parallel_methods[parent.f_code]
                parent = parent.f_back
                assert parent is not None and parent.f_code is self.timed_code
                assert parent.f_locals["self"] is target
                func = parent.f_locals["func"]
                assert (
                    type(func) is MethodType
                    and func.__self__ is target
                    and func.__func__ is callback
                )
                parent = parent.f_back
                assert parent is not None and parent.f_code is self.work_code
                item = parent.f_locals["self"]
                assert type(item) is _WorkItem and type(item.future) is Future
                assert (
                    type(item.fn) is MethodType
                    and item.fn.__self__ is target
                    and item.fn.__func__ is self.timed
                )
                assert item.future.running() and not item.future.done()
                assert threading.current_thread() is not self.creator
                if not any(row[1] is item.future for row in self.parallel_producers):
                    assert len(self.parallel_producers) < 3
                    self.parallel_producers.append(
                        (threading.current_thread(), item.future)
                    )
            else:
                return
            assert len(self.births) < 16 and len(self.owners) < 16
            self.births[id(frame)] = (code, owner, expected)
        except BaseException as error:
            self.invalid.append(type(error).__name__ + ":creator_start")
        finally:
            del frame, parent, owner, item

    def _return(self, code, _offset, value):
        frame = owner = participant = None
        try:
            frame = sys._getframe(1)
            if code is self.app_code:
                assert (
                    id(frame) == self.app_frame
                    and frame.f_locals.get("self") is self.target
                )
                assert value is None and threading.current_thread() is self.creator
                self.app_returned = True
                self.app_frame = None
                return
            birth = self.births.pop(id(frame), None)
            if birth is None:
                return
            original_code, owner, expected = birth
            assert code is original_code and frame.f_code is code
            assert frame.f_locals.get("self") is owner and type(owner) is expected
            assert value is None
            if owner.is_memory_db:
                return
            path = Path(owner.db_path)
            assert path.is_absolute() and path.resolve().is_relative_to(self.root)
            from tldw_chatbook.Backup_Recovery.participants import (
                _repository_participant,
            )

            participant = _repository_participant(owner)
            assert participant.repository() is owner and participant.path == path
            self.owners.append(
                (owner, expected, path, participant, tuple(participant.connections))
            )
        except BaseException as error:
            self.invalid.append(type(error).__name__ + ":creator_return")
        finally:
            del frame, owner, participant, value

    def finish_construction(self, app):
        assert (
            self.target is app
            and self.app_returned
            and not self.births
            and not self.invalid
        )
        assert all(
            future.done() and not future.cancelled()
            for _thread, future in self.parallel_producers
        )
        self.receipt = self.close()
        assert self.receipt["complete"] and self.receipt["original_source_current"]
        assert self.receipt["hooks_retired_before_inactive"]
        assert self.owners

    def retire(self):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from tldw_chatbook.Backup_Recovery.participants import _close_settled_core_cache

        assert threading.current_thread() is self.creator and not self.retired
        for owner, expected, path, participant, initial_connections in self.owners:
            assert type(owner) is expected and owner.db_path == path
            assert participant.repository() is owner and participant.path == path
            with storage._lock:
                settled_closed = not participant.connections
                assert not participant.retiring_threads and storage._pause is None
                active = [
                    op for op in storage._operations if op.participant is participant
                ]
                assert not active, {
                    "unretired_constructor_owner": participant.owner_id,
                    "active_threads": [op.thread.name for op in active],
                    "active_task_types": [type(op.task).__name__ for op in active],
                }
                assert not any(
                    getattr(a.operation, "participant", None) is participant
                    for a in storage._pending_acquisitions
                )
            closed = settled_closed or _close_settled_core_cache(owner)
            if not closed:
                with storage._lock:
                    remaining = tuple(participant.connections.items())
                handles = []
                for connection, lease in remaining:
                    try:
                        transaction = sqlite3.Connection.in_transaction.__get__(
                            connection
                        )
                    except sqlite3.ProgrammingError:
                        transaction = "closed_or_foreign_thread"
                    handles.append(
                        {
                            "thread": getattr(lease.resource_thread, "name", None),
                            "transaction": transaction,
                            "live_lease": lease in storage._live_leases,
                        }
                    )
                raise AssertionError(
                    {
                        "fixture_creator_not_settled": participant.owner_id,
                        "participant_closed": participant.closed,
                        "handles": handles,
                    }
                )
            for connection in initial_connections:
                try:
                    sqlite3.Connection.in_transaction.__get__(connection)
                except sqlite3.ProgrammingError:
                    pass
                else:
                    raise AssertionError("fixture_original_native_creator_not_closed")
            with storage._lock:
                assert not participant.connections and not participant.retiring_threads
                assert not any(
                    op.participant is participant for op in storage._operations
                )
                assert not any(
                    getattr(a.operation, "participant", None) is participant
                    for a in storage._pending_acquisitions
                )
                assert not any(
                    lease.resource_path == path for lease in storage._live_leases
                )
        self.retired = True
        return len(self.owners)
