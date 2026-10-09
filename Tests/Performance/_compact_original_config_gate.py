"""Hold one original checked settings body in compose/on_mount; no replacement."""

import asyncio
import ast
import inspect
import os
import sys
from pathlib import Path
import threading
import time
from types import ModuleType

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver


def coordinate_fixture_editor(source, selected, deadline):
    """Publish only the fixture comment under original finite config custody."""
    from _thread import RLock
    from contextlib import ExitStack
    import hashlib
    import time
    from tldw_chatbook.Backup_Recovery import (
        config_participants,
        raw_participants,
        storage_admission,
    )
    from tldw_chatbook.Utils import platform_files, private_paths

    assert source._get_effective_config_path() == selected
    names = (
        "_CONFIG_CACHE",
        "_CONFIG_CACHE_SOURCE",
        "_SETTINGS_CACHE",
        "_SETTINGS_CACHE_SOURCE",
        "_SETTINGS_CACHE_POSTURE",
        "_CONFIG_GENERATION",
    )
    locks = source._settings_rebuild_lock(), source._config_file_lock()
    assert all(type(lock) is RLock and not lock._is_owned() for lock in locks)
    active = None
    leases = ()
    comment = "\n# compact selected reader external editor\n"
    with ExitStack() as owned:
        for lock in locks:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and lock.acquire(
                timeout=remaining
            ), "compact_editor_lock_deadline"
            owned.callback(lock.release)
        before_cache = tuple(vars(source).get(name) for name in names)
        original = selected.read_text(encoding="utf-8")
        before = platform_files.os.stat(selected, follow_symlinks=False)
        with config_participants.operation(source, wait_for_locks=False) as active:
            state = raw_participants._states[active]
            assert state.source is source and state.selected == selected
            leases = tuple(state.leases)
            result = private_paths.atomic_private_write_text(
                selected, original + comment
            )
            assert result.lexical_path == selected
        assert active not in raw_participants._states
        after = platform_files.os.stat(selected, follow_symlinks=False)
        actual = selected.read_text(encoding="utf-8")
        assert actual == original + comment
        assert (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        assert all(
            vars(source).get(name) is value for name, value in zip(names, before_cache)
        )
    assert all(not lock._is_owned() for lock in locks)
    assert all(lease not in storage_admission._live_leases for lease in leases)
    return {
        "same_editor_deadline_preserved": time.monotonic() <= deadline,
        "body_before_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        "body_after_sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
        "body_is_original_plus_only_fixture_comment": True,
        "raw_file_identity_changed": True,
        "cache_mapping_source_and_generation_unchanged": True,
        "original_config_locks_retired": True,
        "exact_raw_operation_and_leases_retired": True,
        "guard_or_cache_replacements_installed": False,
    }


class OriginalCompactModelReadGate(OriginalStorageUnitObserver):
    def __init__(self, widget, loop, selected_path, target):
        super().__init__({}, lambda: False, lambda _unit: None)
        assert target in {"compose", "mount"}
        self.widget, self.loop, self.selected, self.target = (
            widget,
            loop,
            Path(selected_path).absolute(),
            target,
        )
        self.main_thread = threading.current_thread()
        self.compose_code = self.mount_code = self.reader_code = self.body_code = None
        self.edit = threading.Event()
        self.edited = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.progress = threading.Event()
        self.body_returned = False
        self.controller = None
        self.operation = None
        self.leases = ()
        self.scope_facts = None
        self.controller_stage = "not_started"
        self.target_start_seen = False
        self.worker_code = None
        self.worker_request = None
        self.owned_reads = {}
        self.worker_returned = False
        self.ancestry_route = None
        self.eligibility_codes = {}
        self.eligibility_rows = []
        self.eligibility_seen = set()
        self.eligibility_overflow = 0
        self.capture_return_count = 0
        self.capture_start_count = 0
        self.sidebar_code = None
        self.public_reader_selected_calls = 0
        self.reader_prelude_seen = False
        self.reader_prelude_request = None
        self.editor_deadline = None
        self.editor_facts = None
        self.editor_error = None
        self.loop_stage_sources = {}
        self.external_caller_sources = {}
        self.loop_stage_rows = {}
        self.loop_stage_overflow = 0
        self.loop_probe_queued_at = None
        self.loop_probe_called_at = None
        self.loop_probe_release_at = None
        self.loop_probe_called_after_release = None
        self.local_timeout_stack_lead = []

    def _ancestry(self, frame, wanted):
        found = set()
        for _ in range(16):
            if frame is None:
                break
            if frame.f_code is wanted:
                if frame.f_locals.get("self") is not self.widget:
                    return False
                found.add(wanted)
            if frame.f_code is self.reader_code:
                assert frame.f_globals is self.config.__dict__
                found.add(self.reader_code)
            frame = frame.f_back
        return {wanted, self.reader_code} <= found

    def _worker_ancestry(self, frame):
        if self.worker_code is None:
            return None
        found = set()
        request = None
        for _ in range(16):
            if frame is None:
                break
            if frame.f_code is self.worker_code:
                assert frame.f_globals is self.worker_namespace
                request = frame.f_locals.get("request")
                if (
                    type(request) is not self.setup_type
                    or request.widget is not self.widget
                ):
                    return None
                assert request.app is self.widget.app_instance
                assert request.loop is self.loop and request.thread is self.main_thread
                assert request.reader is self.config.get_cli_providers_and_models
                self._qualify_issued_read(request, frame)
                found.add(self.worker_code)
            if frame.f_code is self.reader_code:
                assert frame.f_globals is self.config.__dict__
                found.add(self.reader_code)
            frame = frame.f_back
        return request if {self.worker_code, self.reader_code} <= found else None

    def _qualify_issued_read(self, request, frame):
        from functools import partial
        from tldw_chatbook.Chat import console_preparation_reads as preparation

        assert (
            type(request.task) is asyncio.Task and request.task.get_loop() is self.loop
        )
        reads = preparation.preparation_reads_for(request.reads)
        assert len(reads) == 1
        read = reads[0]
        assert type(read) is preparation.ConsolePreparationRead
        assert read.creator is self.widget and read.source is request
        assert read.task is request.task and read.session_id is None
        assert read.callback is request.callback and type(read.callback) is partial
        assert read.callback.func is self.worker_function
        assert read.callback.args == (request,) and not read.callback.keywords
        assert read in request.runtime_reads
        assert read._producer is not None and read._producer.get_loop() is self.loop
        assert not read._producer.done() and not read.retired.done()
        assert preparation.preparation_source_for(self.widget) is request
        for _ in range(16):
            if frame is None:
                break
            if frame.f_code is self.preparation_invoke_code:
                assert frame.f_globals is vars(preparation)
                assert frame.f_locals.get("read") is read
                assert frame.f_locals.get("callback") is request.callback
                assert frame.f_locals.get("submitted") is True
                self.owned_reads[id(request)] = read
                return read
            frame = frame.f_back
        raise AssertionError("original_private_producer_callback_not_found")

    def _physically_retired(self, request):
        if request is None:
            return True
        read = self.owned_reads.get(id(request))
        return (
            read is not None
            and read.source is request
            and read.retired.done()
            and not read.retired.cancelled()
            and read._producer.done()
            and not read._producer.cancelled()
            and read not in request.reads
            and read not in request.runtime_reads
            and request.finished
        )

    def _capture_context(self, frame):
        for _ in range(20):
            if frame is None:
                break
            if frame.f_code is self.capture_code:
                assert frame.f_globals is self.worker_namespace
                return frame if frame.f_locals.get("widget") is self.widget else None
            frame = frame.f_back
        return None

    def _eligibility_return(self, code, value):
        frame = self._frame(code)
        role = self.eligibility_codes[code]
        request = frame.f_locals.get("request")
        context = (
            frame
            if role == "owner" and getattr(request, "widget", None) is self.widget
            else self._capture_context(frame)
        )
        if context is None:
            return
        assert frame.f_globals is self.worker_namespace
        role = self.eligibility_codes[code]
        row = {
            "role": role,
            "return_line": frame.f_lineno,
            "returned_none": value is None,
            "returned_false": value is False,
        }
        if role == "owner":
            local = frame.f_locals
            fields, app_fields, runtime_fields = (
                local.get("fields"),
                local.get("app_fields"),
                local.get("runtime_fields"),
            )
            parent, current = local.get("parent"), local.get("current")
            row.update(
                cancelled=request.cancelled,
                reached_owner_fields="fields" in local,
                widget_fields_current=fields is request.widget_fields,
                app_fields_current=app_fields is request.app_fields,
                runtime_fields_current=runtime_fields is request.runtime_fields,
                parent_reference_current=parent is not None
                and parent[0] is request.parent_reference,
                parent_current=parent is not None and parent[1] is request.parent,
                mapping_current=app_fields is not None
                and app_fields.get("app_config") is request.mapping,
                inputs_current=current is not None
                and current[0] is request.inputs[0]
                and current[1] is request.inputs[1]
                and current[2:] == request.inputs[2:],
                identity_current=self.config.current_config_identity()
                == request.identity,
                captured_generation=request.identity[0],
                current_generation=self.config._CONFIG_GENERATION,
                selected_path_current=self.config.current_config_identity()[1]
                == request.identity[1],
                widget_closing=fields is not None and fields.get("_closing") is False,
                widget_closed=fields is not None and fields.get("_closed") is False,
                widget_pruning=fields is not None and fields.get("_pruning") is False,
                request_current=fields is not None
                and fields.get("_compact_setup_request") is request,
                app_running=app_fields is not None and app_fields.get("_exit") is False,
                runtime_live=runtime_fields is not None
                and runtime_fields.get("_disposed") is False,
            )
        if role == "capture":
            self.capture_return_count += 1
        if role == "plain_fields":
            owner = frame.f_locals.get("owner")
            receiver = frame.f_locals.get("receiver")
            names = frame.f_locals.get("names")
            row["receiver_is_target_widget"] = receiver is self.widget
            row["owner_is_original_widget_class"] = owner is self.widget_type
            row["names_include_parent"] = type(names) is tuple and "_parent" in names
            if owner is self.widget_type:
                descriptor = inspect.getattr_static(owner, "_parent", None)
                row["parent_is_property"] = type(descriptor) is property
                row["parent_is_stock_MessagePump_descriptor"] = (
                    descriptor is self.parent_descriptor
                )
                row["stock_getter_body_defaults_closure_namespace"] = (
                    self._parent_current()
                )
        if role == "source":
            source = frame.f_locals.get("source")
            row["capsule_is_original_config"] = source is self.config_capsule
            row["capsule_is_original_widget"] = source is self.widget_capsule
        if role == "function":
            record = frame.f_locals.get("record")
            known = next(
                (
                    label
                    for original, label in self.function_labels
                    if original is record
                ),
                None,
            )
            row["record_is_known_defining_capsule"] = known is not None
            if known is not None:
                row["record_label"] = known
        del frame, context, value
        signature = repr(sorted(row.items()))
        if signature not in self.eligibility_seen:
            if len(self.eligibility_rows) >= 48:
                self.eligibility_overflow += 1
                return
            self.eligibility_seen.add(signature)
            self.eligibility_rows.append(row)

    def _parent_current(self):
        getter, code, namespace, defaults, keywords, closure = self.parent_getter
        return (
            getter.__code__ is code
            and getter.__globals__ is namespace
            and getter.__defaults__ is defaults
            and getter.__kwdefaults__ is keywords
            and getter.__closure__ is closure
            and inspect.getattr_static(self.message_pump_type, "_parent")
            is self.parent_descriptor
        )

    def _edit_before_reader(self, request=None):
        # Fixture input only: the exact reader is paused before its own cache
        # check, while the existing controller performs an ordinary file edit.
        # No settings cache, generation, callback or permission guard changes.
        if self.reader_prelude_seen:
            return
        self.reader_prelude_seen = True
        self.reader_prelude_request = request
        self.editor_deadline = time.monotonic() + 10
        self.edit.set()
        assert self.edited.wait(10)

    def _observe_main_stage(self, code, event, line=None):
        # Passive selected-source scalars only, while the original exact worker
        # is held. No extra read/lock attempt or callable replacement.
        if (
            self.scope_facts is None
            or self.release.is_set()
            or threading.current_thread() is not self.main_thread
        ):
            return
        try:
            label, namespace = self.loop_stage_sources[code]
            frame = self._frame(code)
            assert frame is not None and frame.f_globals is namespace
            if (
                code is self.operation_stage_code
                and frame.f_locals.get("source") is not self.config
            ):
                return
            if (
                code is self.wrapper.__code__
                and frame.f_locals.get("function") is not self.body
            ):
                return
            if event == "line" and line not in self.lock_stage_lines:
                return
            lock = (
                frame.f_locals.get("lock")
                if code is self.operation_stage_code
                else None
            )
            lock_kind = (
                "rebuild"
                if lock is vars(self.config).get("_SETTINGS_REBUILD_LOCK")
                and lock is not None
                else "file"
                if lock is vars(self.config).get("_CONFIG_FILE_LOCK")
                and lock is not None
                else None
            )
            caller = None
            external_caller = None
            external_chain = []
            external_line = None
            identified_external = None
            parent = frame.f_back
            for _ in range(16):
                if parent is None:
                    break
                known = self.loop_stage_sources.get(parent.f_code)
                if known is not None and parent.f_globals is known[1]:
                    if caller is None:
                        caller = known[0]
                external = self.external_caller_sources.get(parent.f_code)
                if external is not None and parent.f_globals is external[1]:
                    if external_caller is None:
                        external_caller, external_line = external[0], parent.f_lineno
                    external_chain.append((external[0], parent.f_lineno))
                # Identification only: never treat an unknown live caller as a
                # defining-source qualification or retain its frame/namespace.
                if identified_external is None:
                    parent_namespace = parent.f_globals
                    name = parent_namespace.get("__name__")
                    if (
                        type(name) is str  # noqa: E721 -- identify only exact source scalars
                        and name.startswith("tldw_chatbook.")
                        and name
                        not in (
                            "tldw_chatbook.config",
                            "tldw_chatbook.Backup_Recovery.config_participants",
                        )
                    ):
                        module = sys.modules.get(name)
                        qualname = parent.f_code.co_qualname
                        if (
                            type(module) is ModuleType
                            and vars(module) is parent_namespace
                            and type(qualname) is str  # noqa: E721 -- exact source scalar
                        ):
                            identified_external = (
                                name[:180],
                                qualname[:180],
                                parent.f_lineno,
                            )
                parent = parent.f_back
            key = (
                label,
                event,
                frame.f_lineno,
                lock_kind,
                caller,
                external_caller,
                external_line,
                identified_external,
                tuple(external_chain),
            )
            now = time.monotonic()
            if key in self.loop_stage_rows:
                row = self.loop_stage_rows[key]
                row["last"] = now
                row["events"] += 1
            elif len(self.loop_stage_rows) < 32:
                self.loop_stage_rows[key] = {
                    "source": label,
                    "event": event,
                    "line": frame.f_lineno,
                    "lock": lock_kind,
                    "selected_original_caller": caller,
                    "external_original_caller": external_caller,
                    "external_original_line": external_line,
                    "external_caller_qualified": external_caller is not None,
                    "qualified_external_ancestry": tuple(external_chain),
                    "identified_external_module": (
                        identified_external[0] if identified_external else None
                    ),
                    "identified_external_qualname": (
                        identified_external[1] if identified_external else None
                    ),
                    "identified_external_line": (
                        identified_external[2] if identified_external else None
                    ),
                    "identified_external_source_qualified": False,
                    "first": now,
                    "last": now,
                    "events": 1,
                    "actual_Main_Thread_while_exact_worker_held": True,
                }
            else:
                self.loop_stage_overflow += 1
            del frame, parent
        except Exception as error:
            self.invalid.append("compact_main_stage:" + type(error).__name__)

    def _line(self, code, line):
        if self.active and code is self.operation_stage_code:
            self._observe_main_stage(code, "line", line)

    def _probe(self):
        self.loop_probe_called_at = time.monotonic()
        self.loop_probe_called_after_release = self.release.is_set()
        if not self.release.is_set():
            self.progress.set()

    def _control(self):
        try:
            self.controller_stage = "waiting_original_target_start"
            assert self.edit.wait(10)
            if self.editor_deadline is None:
                return  # close() wakes an unarmed observer only to retire it.
            self.controller_stage = "original_config_editor_custody"
            self.editor_facts = coordinate_fixture_editor(
                self.config, self.selected, self.editor_deadline
            )
            assert self.editor_facts["same_editor_deadline_preserved"]
            self.edited.set()
            self.controller_stage = "waiting_original_checked_body"
            assert self.entered.wait(10)
            self.controller_stage = "actual_loop_probe"
            self.loop_probe_queued_at = time.monotonic()
            self.loop.call_soon_threadsafe(self._probe)
            progressed = self.progress.wait(0.1)
            if not progressed and os.environ.get("TLDW_LOCAL_COMPACT_STACKS") == "1":
                # Local-only diagnostic lead after the unchanged response bound.
                # Keep code names/lines only, never locals or frame references.
                frame = sys._current_frames().get(self.main_thread.ident)
                try:
                    for _ in range(32):
                        if frame is None:
                            break
                        self.local_timeout_stack_lead.append(
                            (
                                Path(frame.f_code.co_filename).name,
                                frame.f_code.co_qualname,
                                frame.f_lineno,
                            )
                        )
                        frame = frame.f_back
                finally:
                    del frame
        except BaseException as error:
            self.editor_error = {
                "phase": self.controller_stage,
                "type": type(error).__name__,
                "errno": error.errno
                if isinstance(error, OSError) and type(error.errno) is int  # noqa: E721 -- bounded scalar only
                else None,
                "winerror": getattr(error, "winerror", None)
                if type(getattr(error, "winerror", None)) is int  # noqa: E721 -- bounded scalar only
                else None,
            }
            self.invalid.append("compact_controller:" + type(error).__name__)
        finally:
            self.edited.set()
            self.loop_probe_release_at = time.monotonic()
            self.release.set()

    def _start(self, code, offset):
        if self.active and code in self.loop_stage_sources:
            self._observe_main_stage(code, "start")
        if self.active and code in self.eligibility_codes:
            if code is self.capture_code:
                frame = self._frame(code)
                if frame.f_locals.get("widget") is self.widget:
                    assert frame.f_globals is self.worker_namespace
                    self.capture_start_count += 1
                del frame
            return
        if self.active and code is self.reader_code:
            frame = self._frame(code)
            if self._ancestry(frame, self.sidebar_code):
                self.public_reader_selected_calls += 1
            if self.target == "compose":
                try:
                    request = self._worker_ancestry(frame)
                    if self._ancestry(frame, self.compose_code) or request is not None:
                        self._edit_before_reader(request)
                except BaseException as error:
                    self.invalid.append(
                        "compact_reader_prelude:" + type(error).__name__
                    )
                    self.release.set()
            del frame
        if self.active and code is self.mount_code and self.target == "mount":
            try:
                frame = self._frame(code)
                if frame.f_locals.get("self") is not self.widget:
                    return
                assert frame.f_globals is self.mount_namespace
                self.target_start_seen = True
                self.editor_deadline = time.monotonic() + 10
                self.edit.set()
                assert self.edited.wait(10)
            except BaseException as error:
                self.invalid.append("compact_mount_prelude:" + type(error).__name__)
                self.release.set()
            return
        if (
            not self.active
            or code is not self.body_code
            or self.scope_facts is not None
        ):
            return
        try:
            frame = self._frame(code)
            wanted = self.compose_code if self.target == "compose" else self.mount_code
            original_main = self._ancestry(frame, wanted)
            request = None if original_main else self._worker_ancestry(frame)
            if not original_main and request is None:
                return
            current_thread = threading.current_thread()
            if original_main:
                assert current_thread is self.main_thread
                assert asyncio.get_running_loop() is self.loop
                current_task = asyncio.current_task()
                self.ancestry_route = "original_main"
            else:
                assert current_thread is not self.main_thread
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    current_task = None
                else:
                    raise AssertionError("compact_worker_has_UI_loop")
                self.worker_request = request
                self.ancestry_route = "issued_stock_worker"
            assert frame.f_globals is self.config.__dict__
            parent = frame.f_back
            assert parent.f_code is self.wrapper.__code__
            assert parent.f_globals is self.participants.__dict__
            assert parent.f_locals.get("function") is self.body
            operation = getattr(self.raw._local, "operation", None)
            assert operation is not None
            state = self.raw._states[operation]
            assert state.source is self.config and state.selected == self.selected
            assert state.active and state.pinned and not state.uncertain
            assert state.thread is current_thread and state.task is current_task
            assert state.pid == os.getpid() and state.participant is not None
            assert state.leases and all(
                lease in self.storage._live_leases for lease in state.leases
            )
            assert len(state.leases) == len(state.holds)
            assert all(
                self.storage._holds.get(lease._key) is hold
                for lease, hold in zip(state.leases, state.holds)
            )
            self.operation, self.leases = operation, tuple(state.leases)
            self.scope_facts = dict(
                original_widget_reader_ancestry=True,
                actual_checked_raw_scope=True,
                actual_scope_thread_and_task=True,
                actual_live_storage_leases=True,
                config_generation=self.config._CONFIG_GENERATION,
                operation_actor=id(operation),
                widget_actor=id(self.widget),
                ancestry_route=self.ancestry_route,
                actual_UI_Task_issued_stock_worker=request is not None,
                checked_scope_on_UI_thread=current_thread is self.main_thread,
            )
            self.entered.set()
            assert self.release.wait(10)
        except BaseException as error:
            self.invalid.append("compact_start:" + type(error).__name__)
            self.entered.set()
            self.release.set()

    def _return(self, code, offset, value):
        if self.active and code in self.loop_stage_sources:
            self._observe_main_stage(code, "return")
        if not self.active:
            return
        try:
            if code in self.eligibility_codes:
                self._eligibility_return(code, value)
                return
            frame = self._frame(code)
            if code is self.worker_code and self.worker_request is not None:
                if frame.f_locals.get("request") is self.worker_request:
                    assert frame.f_globals is self.worker_namespace
                    self.worker_returned = True
            if code is self.body_code and self.scope_facts is not None:
                if (
                    self._worker_ancestry(frame) is self.worker_request
                    and self.worker_request is not None
                ):
                    self.body_returned = True
                elif self._ancestry(
                    frame,
                    self.compose_code if self.target == "compose" else self.mount_code,
                ):
                    self.body_returned = True
        except BaseException as error:
            self.invalid.append("compact_return:" + type(error).__name__)
            self.release.set()

    def install(self):
        from tldw_chatbook import config
        from tldw_chatbook.Backup_Recovery import (
            config_participants,
            raw_participants,
            storage_admission,
        )
        from tldw_chatbook.Widgets import compact_model_bar

        # These helpers define the exact selected Main/issued-worker route.
        for name in ("_ancestry", "_worker_ancestry", "_edit_before_reader"):
            function = inspect.getattr_static(type(self), name)
            self._pin(function)
            self.slots.append((type(self), name, function))
        self.config, self.participants, self.raw, self.storage = (
            config,
            config_participants,
            raw_participants,
            storage_admission,
        )
        from tldw_chatbook.Utils import private_paths

        for function in (
            coordinate_fixture_editor,
            config._settings_rebuild_lock,
            config._config_file_lock,
            private_paths.atomic_private_write_text,
            private_paths.atomic_private_write_bytes,
        ):
            self._pin(function)
            if hasattr(function, "__wrapped__"):
                self._pin(inspect.getattr_static(function, "__wrapped__"))
        for owner, name in (
            (sys.modules[__name__], "coordinate_fixture_editor"),
            (config, "_settings_rebuild_lock"),
            (config, "_config_file_lock"),
            (private_paths, "atomic_private_write_text"),
            (private_paths, "atomic_private_write_bytes"),
        ):
            self.slots.append((owner, name, inspect.getattr_static(owner, name)))
        assert type(self.widget) is compact_model_bar.CompactModelBar
        assert (
            compact_model_bar.get_cli_providers_and_models
            is config.get_cli_providers_and_models
        )
        self.wrapper = config._load_settings_guarded
        self.body = inspect.getattr_static(self.wrapper, "__wrapped__")
        cells = dict(zip(self.wrapper.__code__.co_freevars, self.wrapper.__closure__))
        assert (
            cells["function"].cell_contents is self.body
            and cells["wrapped"].cell_contents is self.wrapper
        )
        for owner, name, label in (
            (compact_model_bar.CompactModelBar, "compose", "compose_code"),
            (compact_model_bar.CompactModelBar, "on_mount", "mount_code"),
            (config, "get_cli_providers_and_models", "reader_code"),
        ):
            function = inspect.getattr_static(owner, name)
            setattr(self, label, self._pin(function))
            if label == "mount_code":
                self.mount_namespace = function.__globals__
            self.slots.append((owner, name, function))
        self._pin(self.wrapper)
        self.body_code = self._pin(self.body)
        self.slots.extend(
            (
                (config, "_load_settings_guarded", self.wrapper),
                (self.wrapper, "__wrapped__", self.body),
                (
                    compact_model_bar,
                    "get_cli_providers_and_models",
                    config.get_cli_providers_and_models,
                ),
            )
        )
        for owner, name in (
            (config, "load_settings"),
            (self.participants, "operation"),
            (self.raw, "_scope"),
        ):
            function = inspect.getattr_static(owner, name)
            self._pin(function)
            self.slots.append((owner, name, function))
            if hasattr(function, "__wrapped__"):
                body = inspect.getattr_static(function, "__wrapped__")
                self._pin(body)
                self.slots.append((function, "__wrapped__", body))
        worker = compact_model_bar.__dict__.get("_read_compact_setup")
        if worker is not None:
            self.worker_function = worker
            self.worker_code = self._pin(worker)
            self.worker_namespace = worker.__globals__
            self.setup_type = compact_model_bar._CompactSetup
            from tldw_chatbook.Chat import console_preparation_reads as preparation
            from types import CodeType

            self._pin(preparation.run_preparation_read)
            self._pin(preparation.preparation_source_for)
            self.slots.extend(
                (
                    (
                        preparation,
                        "run_preparation_read",
                        preparation.run_preparation_read,
                    ),
                    (
                        preparation,
                        "preparation_source_for",
                        preparation.preparation_source_for,
                    ),
                )
            )
            invokes = tuple(
                code
                for code in preparation.run_preparation_read.__code__.co_consts
                if type(code) is CodeType and code.co_name == "invoke"
            )
            assert len(invokes) == 1
            self.preparation_invoke_code = invokes[0]
            self.slots.extend(
                (
                    (compact_model_bar, "_read_compact_setup", worker),
                    (compact_model_bar, "_CompactSetup", self.setup_type),
                )
            )
        self.codes = {
            self.compose_code: "original_compose",
            self.mount_code: "original_mount",
            self.body_code: "original_checked_settings_body",
        }
        if self.worker_code is not None:
            self.codes[self.worker_code] = "issued_stock_compact_worker"
        from textual.message_pump import MessagePump

        self.widget_type = compact_model_bar.CompactModelBar
        self.message_pump_type = MessagePump
        self.parent_descriptor = inspect.getattr_static(MessagePump, "_parent")
        assert type(self.parent_descriptor) is property
        getter = self.parent_descriptor.fget
        self._pin(getter)
        assert getter.__closure__ is None and getter.__globals__ is vars(
            __import__("textual.message_pump", fromlist=["MessagePump"])
        )
        self.parent_getter = (
            getter,
            getter.__code__,
            getter.__globals__,
            getter.__defaults__,
            getter.__kwdefaults__,
            getter.__closure__,
        )
        self.slots.append((MessagePump, "_parent", self.parent_descriptor))
        self.config_capsule = compact_model_bar._CONFIG_SOURCE
        self.widget_capsule = compact_model_bar._WIDGET_SOURCE
        self.function_labels = tuple(
            (record, role + ":" + record[1])
            for role, capsule in (
                ("config", self.config_capsule),
                ("widget", self.widget_capsule),
            )
            for record in capsule[5]
        )
        for name, role in (
            ("_capture_setup", "capture"),
            ("_setup_owner_current", "owner"),
            ("_sources_current", "sources"),
            ("_source_current", "source"),
            ("_function_current", "function"),
            ("_plain_fields", "plain_fields"),
            ("_setup_inputs", "inputs"),
        ):
            function = vars(compact_model_bar)[name]
            code = self._pin(function)
            self.slots.append((compact_model_bar, name, function))
            self.eligibility_codes[code] = role
            self.codes[code] = "original_compact_eligibility_" + role
            if role == "capture":
                self.capture_code = code
        sidebar = inspect.getattr_static(self.widget_type, "sync_from_sidebar")
        self.sidebar_code = self._pin(sidebar)
        self.slots.append((self.widget_type, "sync_from_sidebar", sidebar))
        self.codes[self.sidebar_code] = "original_automatic_or_direct_sidebar_sync"
        self.codes[self.reader_code] = "original_public_reader"
        # Only already-loaded defining functions. No App/service imports and
        # no global exception/native-work event set.
        operation = inspect.getattr_static(self.participants, "operation")
        generator = inspect.getattr_static(operation, "__wrapped__")
        self.operation_stage_code = self._pin(generator)
        self.slots.append((operation, "__wrapped__", generator))
        self.lock_stage_lines = frozenset(
            node.lineno
            for node in ast.walk(
                ast.parse(Path(self.participants.__file__).read_bytes())
            )
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "acquire"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "lock"
        )
        for function, label in (
            (config.load_settings, "config.load_settings"),
            (self.wrapper, "config._load_settings_guarded.wrapper"),
            (generator, "config.operation.generator"),
            (config._settings_cache_hit, "config._settings_cache_hit"),
        ):
            code = self._pin(function)
            self.loop_stage_sources[code] = label, function.__globals__
            self.codes[code] = "original_compact_Main_stage"
        for code, label, namespace in (
            (self.reader_code, "config.get_cli_providers_and_models", vars(config)),
            (self.sidebar_code, "compact.sync_from_sidebar", self.worker_namespace),
            (self.mount_code, "compact.on_mount", self.mount_namespace),
        ):
            self.loop_stage_sources[code] = label, namespace
        self.slots.extend(
            (
                (config, "_settings_cache_hit", config._settings_cache_hit),
                (self.participants, "operation", operation),
            )
        )
        # Qualify concrete already-loaded consumers before the held callback.
        # They need no event mask: only an actual selected frame's ancestry is
        # inspected. Unknown callers remain explicitly unavailable.
        from types import ModuleType

        for module_name, class_name, names in (
            (
                "tldw_chatbook.UI.Console_Modules.context_spend",
                "ConsoleContextSpendController",
                (
                    "_build_console_settings_summary_state",
                    "_active_console_settings_context_estimate",
                ),
            ),
            (
                "tldw_chatbook.UI.Screens.chat_screen",
                "ChatScreen",
                (
                    "_sync_compact_shell_controls",
                    "_provider_readiness_app_config",
                    "_sync_console_control_bar",
                    "_run_console_config_sync",
                    "_sync_native_console_chat_ui",
                    "_sync_console_rail_and_controls",
                    "compose_content",
                ),
            ),
            (
                "tldw_chatbook.Widgets.Console.console_transcript",
                "ConsoleTranscript",
                (
                    "_turn_file_cards_enabled",
                    "compose",
                    "refresh_messages",
                    "_reconcile_rows",
                    "_sync_assistant_turn_widget",
                ),
            ),
            (
                "tldw_chatbook.Chat.console_provider_gateway",
                "ConsoleProviderGateway",
                ("cached_context_window",),
            ),
            (
                "tldw_chatbook.Chat.console_context_window",
                "ContextWindowCache",
                ("cached",),
            ),
            (
                "tldw_chatbook.model_capabilities",
                "ModelCapabilities",
                ("get_model_capabilities",),
            ),
            (
                "tldw_chatbook.LLM_Calls.pricing_catalog",
                "PricingCatalog",
                ("get_pricing",),
            ),
            ("tldw_chatbook.app", "TldwCli", ("compose", "on_mount")),
            (
                "tldw_chatbook.app_service_wiring",
                "ServiceWiringMixin",
                ("_build_local_skill_trust_service",),
            ),
        ):
            module = sys.modules.get(module_name)
            if type(module) is not ModuleType:
                continue
            owner = vars(module).get(class_name)
            if not isinstance(owner, type):
                continue
            self.slots.append((module, class_name, owner))
            for name in names:
                function = inspect.getattr_static(owner, name)
                code = self._pin(function)
                self.slots.append((owner, name, function))
                self.external_caller_sources[code] = (
                    class_name + "." + name,
                    function.__globals__,
                )
        for module_name, names in (
            ("tldw_chatbook.model_capabilities", ("_models_dev_capabilities",)),
            ("tldw_chatbook.LLM_Calls.pricing_catalog", ("_models_dev_pricing",)),
            ("tldw_chatbook.Utils.token_counter", ("resolve_context_window",)),
            (
                "tldw_chatbook.LLM_Provider_Catalog.models_dev_catalog",
                ("models_dev_entry", "_enabled"),
            ),
        ):
            module = sys.modules.get(module_name)
            if type(module) is not ModuleType:
                continue
            for name in names:
                function = inspect.getattr_static(module, name)
                code = self._pin(function)
                self.slots.append((module, name, function))
                self.external_caller_sources[code] = (
                    module_name.rsplit(".", 1)[-1] + "." + name,
                    function.__globals__,
                )
        runtime = sys.modules.get("tldw_chatbook.Chat.console_runtime")
        if type(runtime) is ModuleType:
            capsule = vars(runtime).get("_PROVIDER_CONFIG_FOR_APP_ORIGINAL")
            assert type(capsule) is tuple and len(capsule) == 3
            namespace, function, code = capsule
            assert namespace is vars(runtime)
            assert namespace.get("_provider_config_for_app") is function
            assert function.__code__ is code and function.__globals__ is namespace
            self._pin(function)
            self.slots.append((runtime, "_provider_config_for_app", function))
            self.external_caller_sources[code] = (
                "console_runtime._provider_config_for_app",
                namespace,
            )
        presentation = sys.modules.get(
            "tldw_chatbook.UI.Console_Modules.console_spend_projection"
        )
        if type(presentation) is ModuleType:
            name = "provider_readiness_app_config"
            function = inspect.getattr_static(presentation, name)
            code = self._pin(function)
            self.slots.append((presentation, name, function))
            self.external_caller_sources[code] = (
                "console_spend_projection." + name,
                function.__globals__,
            )
        for code, label, namespace in (
            (self.compose_code, "CompactModelBar.compose", self.worker_namespace),
            (self.mount_code, "CompactModelBar.on_mount", self.mount_namespace),
            (
                self.sidebar_code,
                "CompactModelBar.sync_from_sidebar",
                self.worker_namespace,
            ),
        ):
            self.external_caller_sources[code] = label, namespace
        for name in ("_observe_main_stage", "_line", "_probe"):
            function = inspect.getattr_static(type(self), name)
            self._pin(function)
            self.slots.append((type(self), name, function))
        for tool in range(5, 0, -1):
            if tool == self.monitor.DEBUGGER_ID:
                continue
            try:
                self.monitor.use_tool_id(tool, "compact-original-config-read")
            except ValueError:
                continue
            self.tool = tool
            break
        assert self.tool is not None and self.monitor.get_events(self.tool) == 0
        for event, callback in (
            (self.monitor.events.PY_START, self._start),
            (self.monitor.events.PY_RETURN, self._return),
            (self.monitor.events.LINE, self._line),
        ):
            previous = self.monitor.register_callback(self.tool, event, callback)
            if previous is not None:
                self.monitor.register_callback(self.tool, event, previous)
                raise AssertionError("compact_callback_borrowed")
            self.registered[event] = callback
        self.active = self.installed = True
        for code in self.codes:
            self.monitor.set_local_events(
                self.tool,
                code,
                self.monitor.events.PY_START
                | self.monitor.events.PY_RETURN
                | (
                    self.monitor.events.LINE if code is self.operation_stage_code else 0
                ),
            )
        self.controller = threading.Thread(
            target=self._control, name="compact-original-read-controller"
        )
        self.controller.start()

    def close(self):
        self.release.set()
        self.edit.set()
        self.edited.set()
        if self.controller is not None:
            self.controller.join(10)
            if self.controller.is_alive():
                self.invalid.append("compact_controller_not_retired")
        result = super().close()
        result.update(
            scope=self.scope_facts,
            original_body_normal_RETURN=self.body_returned,
            original_raw_scope_and_leases_retired=(
                self.operation is not None
                and self.operation not in self.raw._states
                and all(lease not in self.storage._live_leases for lease in self.leases)
            ),
            actual_loop_progress_while_original_checked_read_held=self.progress.is_set(),
            controller_retired=self.controller is None
            or not self.controller.is_alive(),
            target=self.target,
            selected_Main_source_stages=list(self.loop_stage_rows.values()),
            selected_Main_source_stage_overflow=self.loop_stage_overflow,
            actual_loop_probe_queued_at=self.loop_probe_queued_at,
            actual_loop_probe_called_at=self.loop_probe_called_at,
            actual_loop_probe_release_at=self.loop_probe_release_at,
            actual_loop_probe_called_after_release=self.loop_probe_called_after_release,
            selected_Main_stage_is_not_exclusive_time_or_lock_ownership=True,
            editor_facts=self.editor_facts,
            editor_error=self.editor_error,
            selected_original_reader_prelude_seen=self.reader_prelude_seen,
            external_edit_at_selected_reader_before_original_cache_check=self.target
            == "compose"
            and self.reader_prelude_seen,
            selected_prelude_actual_issued_Task_retired=self._physically_retired(
                self.reader_prelude_request
            ),
            exact_original_mount_start_seen=self.target_start_seen,
            controller_last_stage=self.controller_stage,
            worker_original_normal_RETURN=self.worker_returned,
            worker_actual_Task_retired=(self._physically_retired(self.worker_request)),
            known_stock_worker_additive_prerequisite=self.worker_code is not None,
            frames_arguments_results_tasks_retained=self.worker_request is not None,
            exact_known_worker_request_Task_retained_only_for_finite_custody=self.worker_request
            is not None,
            arbitrary_frames_arguments_results_collected=False,
            eligibility_rows=self.eligibility_rows,
            eligibility_overflow=self.eligibility_overflow,
            target_capture_STARTs=self.capture_start_count,
            target_capture_RETURNs=self.capture_return_count,
            target_sidebar_original_public_reader_STARTs=self.public_reader_selected_calls,
            eligibility_observation_is_code_local_scalar_only=True,
            original_eligibility_callbacks_never_replaced=True,
            native_permission_guards_replaced=False,
            configuration_edit_is_fixture_input_not_cache_or_guard_override=True,
        )
        if os.environ.get("TLDW_LOCAL_COMPACT_STACKS") == "1":
            result["local_timeout_stack_lead"] = self.local_timeout_stack_lead
        result["complete"] = (
            result["complete"]
            and not self.eligibility_overflow
            and not self.loop_stage_overflow
            and self.capture_start_count == self.capture_return_count
        )
        return result
