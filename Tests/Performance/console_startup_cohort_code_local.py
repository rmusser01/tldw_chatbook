"""Diagnostic-only code-local observation mechanism for startup cohort metadata.

No global events, audit/profile hooks, guard replacements or native handle ledger.
Successful returned buffers and Native facade HANDLEs are separate API metrics,
not complete kernel I/O. Parent must qualify this before any whole-App use.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import hmac
import importlib.machinery
import inspect
import io
import ntpath
import os
from pathlib import Path
import posixpath
import sys
import threading
import time
from types import CodeType, FunctionType


SPECS = ()  # The scoped witness supplies its exact six selected bodies.


def filename(value):
    value = str(value).replace("\\", "/")
    return value.casefold() if os.name == "nt" else value


def original_body(module_name, member):
    module = sys.modules.get(module_name)
    if module is None:
        return None
    value = module
    for name in member.split("."):
        value = inspect.getattr_static(value, name)
        if type(value) in (classmethod, staticmethod):
            value = value.__func__
    if type(value) is not FunctionType:
        return None
    function = inspect.unwrap(value)
    return (
        (value, function)
        if type(function) is FunctionType and function.__globals__ is vars(module)
        else None
    )


def nested(code, qualname):
    if code.co_qualname == qualname:
        return code
    for value in code.co_consts:
        if type(value) is CodeType:
            found = nested(value, qualname)
            if found is not None:
                return found
    return None


class StartupIOCensus:
    def __init__(self, repo: Path, hash_key: bytes, *, specs=SPECS):
        self.repo, self.hash_key, self.specs = repo.absolute(), hash_key, tuple(specs)
        self.cwd, self.main_ident = os.getcwd(), threading.get_ident()
        self.phase, self.active, self.tool = "python_import", False, None
        self.labels, self.source_specs, self.bindings = {}, {}, {}
        self.local_masks, self.frames, self.publications = {}, {}, {}
        self.rows, self.path_rows, self.issues = Counter(), Counter(), Counter()
        self.first_reads, self.key_events, self.loader_publications = {}, [], []
        self.first_body_entries = {}
        self.expected_screen = self.expected_composer = None
        self.callbacks_after_inactive = 0
        self.callbacks = {}
        self.stock = {
            "path_bytes": inspect.getattr_static(Path, "read_bytes"),
            "path_text": inspect.getattr_static(Path, "read_text"),
            "loader_data": inspect.getattr_static(
                importlib.machinery.SourceFileLoader, "get_data"
            ),
            "loader_code": inspect.getattr_static(
                importlib.machinery.SourceFileLoader, "get_code"
            ),
            "loader_exec": inspect.getattr_static(
                importlib.machinery.SourceFileLoader, "exec_module"
            ),
        }
        assert all(type(body) is FunctionType for body in self.stock.values())

    def _path(self, value):
        if isinstance(value, Path):
            value = str(value)
        if type(value) is bytes:  # noqa: E721 - exact original metadata type.
            value = os.fsdecode(value)
        if type(value) is not str:  # noqa: E721 - exact original metadata type.
            return None
        module = ntpath if os.name == "nt" else posixpath
        if not module.isabs(value):
            value = module.join(self.cwd, value)
        value = module.normpath(value)
        return value.casefold() if os.name == "nt" else value

    def _digest(self, value):
        path = self._path(value)
        return (
            None
            if path is None
            else hmac.new(
                self.hash_key, path.encode("utf-8", "surrogatepass"), hashlib.sha256
            ).hexdigest()
        )

    def _count(self, category, path=None, amount=1):
        thread = "main" if threading.get_ident() == self.main_ident else "worker"
        key = self.phase, thread, category
        self.rows[key] += amount
        if path is not None:
            self.path_rows[(*key, path)] += 1

    def _enable(self, code, label, spec=None):
        if code in self.labels:
            if self.labels[code] != label:
                self.issues["ambiguous_source_code"] += 1
            return
        # Python 3.12 does not permit PY_UNWIND as a local event. Exceptional
        # completions remain explicit count gaps; global events stay exactly 0.
        mask = sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN
        self.labels[code] = label
        if spec is not None:
            self.source_specs[code] = spec
        self.local_masks[code] = mask
        sys.monitoring.set_local_events(self.tool, code, mask)

    def _bind(self, module_name=None):
        for spec in self.specs:
            if (
                spec in self.bindings
                or module_name is not None
                and spec[0] != module_name
            ):
                continue
            try:
                found = original_body(spec[0], spec[1])
            except (AttributeError, ValueError):
                continue
            if found is None:
                continue
            value, body = found
            code = body.__code__ if spec[2] is None else nested(body.__code__, spec[2])
            expected = filename(self.repo / (spec[0].replace(".", "/") + ".py"))
            if code is None or filename(code.co_filename) != expected:
                self.issues["source_filename_mismatch:" + spec[3]] += 1
                continue
            self.bindings[spec] = value, body, code
            self.publications.setdefault(
                code,
                {
                    "phase": self.phase,
                    "time": time.perf_counter(),
                    "kind": "actual_published_function",
                    "label": spec[3],
                },
            )
            self._enable(code, spec[3], spec)

    def _prepare_loaded_code(self, loader, code):
        # Actual original loader-returned module code permits instrumentation
        # before a first top-level call. No synthetic/new code object is made.
        if (
            type(loader) is not importlib.machinery.SourceFileLoader
            or type(code) is not CodeType
        ):
            return
        name = loader.name
        expected = filename(self.repo / (name.replace(".", "/") + ".py"))
        if filename(code.co_filename) != expected:
            return
        for spec in self.specs:
            if spec[0] != name:
                continue
            selected = nested(code, spec[2] or spec[1])
            if selected is not None:
                self._enable(selected, spec[3], spec)

    def _frame(self, frame, code):
        if frame.f_code is not code:
            self.issues["callback_frame_mismatch"] += 1
            return None
        return frame

    def _current_source(self, code):
        spec = self.source_specs.get(code)
        if spec is None:
            return True
        self._bind(spec[0])
        binding = self.bindings.get(spec)
        if binding is None or binding[2] is not code:
            self.issues["first_body_not_published:" + spec[3]] += 1
            return False
        return True

    def _start_event(self, code, offset):
        if not self.active:
            self.callbacks_after_inactive += 1
            return
        try:
            frame = self._frame(sys._getframe(1), code)
            if frame is None:
                return
            self._current_source(code)
            label, values = self.labels[code], frame.f_locals
            self.first_body_entries.setdefault(
                label,
                {"label": label, "time": time.perf_counter(), "phase": self.phase},
            )
            path = None
            if label in ("path_bytes", "path_text"):
                path = self._path(values.get("self"))
            elif label == "loader_data":
                path = self._path(values.get("path"))
            elif label == "native_open" and values.get("parent") is None:
                name = values.get("name")
                # No numeric-HANDLE lifetime guessing, and no relative root name.
                if type(name) is str and ntpath.isabs(name):  # noqa: E721 - exact original metadata type.
                    path = self._path(name)
            elif label in ("samira_read", "pixel_read"):
                path = self._path(values.get("candidate"))
            row = {
                "label": label,
                "path": path,
                "directory": bool(values.get("directory")),
                "metadata": bool(values.get("metadata")),
                "phase": self.phase,
            }
            self.frames[threading.get_ident(), id(frame)] = row
            self._count(label + "_entered", path)
            if label not in ("loader_exec", "loader_code", "loader_data"):
                self.publications.setdefault(
                    code,
                    {
                        "kind": "stock_original",
                        "time": time.perf_counter(),
                        "phase": self.phase,
                        "label": label,
                    },
                )
            if (
                label == "screen_key"
                and values.get("self") is self.expected_screen
                or label in ("composer_insert", "composer_key_dispatch")
                and values.get("self") is self.expected_composer
            ):
                if len(self.key_events) < 8:
                    self.key_events.append(
                        {
                            "source": label,
                            "time": time.perf_counter(),
                            "actor_id": id(values.get("self")),
                        }
                    )
        except Exception as error:
            self.issues["observer_start:" + type(error).__name__] += 1

    def _return_event(self, code, offset, value):
        if not self.active:
            self.callbacks_after_inactive += 1
            return
        try:
            frame = self._frame(sys._getframe(1), code)
            if frame is None:
                return
            row = self.frames.pop((threading.get_ident(), id(frame)), None)
            if row is None:
                self.issues["unmatched_return:" + self.labels[code]] += 1
                return
            label, path = row["label"], row["path"]
            self._count(label + "_returned", path)
            if label == "loader_code":
                self._prepare_loaded_code(frame.f_locals.get("self"), value)
            elif label == "loader_exec":
                module = frame.f_locals.get("module")
                name = getattr(module, "__name__", None)
                if name is not None and sys.modules.get(name) is module:
                    self._bind(name)
                    if any(spec[0] == name for spec in self.specs):
                        self.loader_publications.append(
                            {
                                "module": name,
                                "time": time.perf_counter(),
                                "phase": self.phase,
                                "actor_id": id(module),
                            }
                        )
            elif label == "native_open":
                kind = (
                    "directory"
                    if row["directory"]
                    else "metadata"
                    if row["metadata"]
                    else "file"
                )
                self._count(
                    "native_open_" + kind + "_success"
                    if type(value) is int and value > 0  # noqa: E721 - exact original metadata type.
                    else "native_open_invalid_return",
                    path,
                )
                if path is None:
                    self._count("native_open_unresolved_parent_path")
            elif (
                label in ("path_bytes", "loader_data", "samira_read", "pixel_read")
                and type(value) is bytes  # noqa: E721 - exact original metadata type.
            ):
                if label in ("samira_read", "pixel_read") and path is None:
                    path = self._path(frame.f_locals.get("candidate"))
                    stream = frame.f_locals.get("stream")
                    if path is None and type(stream) in (
                        io.FileIO,
                        io.BufferedReader,
                        io.BufferedRandom,
                        io.TextIOWrapper,
                    ):
                        path = self._path(stream.name)
                self._count(label + "_buffer_success", path)
                self._count(label + "_returned_bytes", amount=len(value))
                key = threading.get_ident(), label, path
                self.first_reads.setdefault(
                    key,
                    {
                        "phase": self.phase,
                        "thread": "main"
                        if threading.get_ident() == self.main_ident
                        else "worker",
                        "source": label,
                        "path": path,
                        "returned_bytes": len(value),
                        "time": time.perf_counter(),
                        "actual_original_return": True,
                    },
                )
            elif label == "path_text" and type(value) is str:  # noqa: E721 - exact original metadata type.
                self._count("path_text_buffer_success", path)
                self._count("path_text_returned_characters", amount=len(value))
        except Exception as error:
            self.issues["observer_return:" + type(error).__name__] += 1

    def start(self):
        assert not self.active and self.tool is None
        self.tool = next(
            index for index in (3, 4) if sys.monitoring.get_tool(index) is None
        )
        self.tool_name = "startup-code-local-" + str(id(self))
        sys.monitoring.use_tool_id(self.tool, self.tool_name)
        assert sys.monitoring.get_events(self.tool) == 0
        self.callbacks = {
            sys.monitoring.events.PY_START: self._start_event,
            sys.monitoring.events.PY_RETURN: self._return_event,
        }
        for event, callback in self.callbacks.items():
            assert sys.monitoring.register_callback(self.tool, event, callback) is None
        self.active = True
        for label, body in self.stock.items():
            self._enable(body.__code__, label)
        self._bind()

    def stop(self):
        # Remain active through mask/callback cleanup. No inactive self-detach.
        assert self.active
        owned = sys.monitoring.get_tool(self.tool) == self.tool_name
        global_zero = sys.monitoring.get_events(self.tool) == 0
        assert owned and global_zero, "Foreign monitoring state; do not overwrite it"
        masks_owned = all(
            sys.monitoring.get_local_events(self.tool, code) == mask
            for code, mask in self.local_masks.items()
        )
        assert masks_owned, "Selected code masks changed"
        identities_stable = all(
            original_body(spec[0], spec[1]) == (value, body)
            for spec, (value, body, code) in self.bindings.items()
        )
        stock_stable = all(
            inspect.getattr_static(
                Path
                if label.startswith("path_")
                else importlib.machinery.SourceFileLoader,
                {
                    "path_bytes": "read_bytes",
                    "path_text": "read_text",
                    "loader_data": "get_data",
                    "loader_code": "get_code",
                    "loader_exec": "exec_module",
                }[label],
            )
            is body
            for label, body in self.stock.items()
        )
        for code in self.local_masks:
            sys.monitoring.set_local_events(self.tool, code, 0)
        masks_cleared = all(
            sys.monitoring.get_local_events(self.tool, code) == 0
            for code in self.local_masks
        )
        callbacks_owned = True
        for event, callback in self.callbacks.items():
            actual = sys.monitoring.register_callback(self.tool, event, None)
            if actual is not callback:
                callbacks_owned = False
                sys.monitoring.register_callback(self.tool, event, actual)
        if callbacks_owned:
            sys.monitoring.free_tool_id(self.tool)
        self.active = False
        tool_freed = sys.monitoring.get_tool(self.tool) is None
        operations = [
            {"phase": p, "thread": t, "operation": op, "count": count}
            for (p, t, op), count in sorted(self.rows.items())
        ]
        path_operations = [
            {
                "phase": p,
                "thread": t,
                "operation": op,
                "path_hmac_sha256": self._digest(path),
                "count": count,
            }
            for (p, t, op, path), count in sorted(self.path_rows.items())
        ]
        first_reads = [
            {
                **{k: v for k, v in row.items() if k != "path"},
                "path_hmac_sha256": self._digest(row["path"]),
            }
            for row in self.first_reads.values()
        ]
        gaps = []
        for (phase, thread, category), count in self.rows.items():
            if category.endswith("_entered"):
                label = category.removesuffix("_entered")
                missing = count - self.rows[phase, thread, label + "_returned"]
                if missing:
                    gaps.append(
                        {
                            "phase": phase,
                            "thread": thread,
                            "source": label,
                            "started_without_same_phase_normal_return": missing,
                        }
                    )
        return {
            "diagnostic_only": True,
            "operations": operations,
            "path_operations": path_operations,
            "unique_hashed_paths": len(
                {row["path_hmac_sha256"] for row in path_operations}
            ),
            "first_actual_read_events": first_reads,
            "actual_original_key_events": self.key_events,
            "actual_module_publications": self.loader_publications,
            "actual_function_publications": list(self.publications.values()),
            "first_selected_body_entries": list(self.first_body_entries.values()),
            "bound_original_bodies": [spec[3] for spec in self.bindings],
            "missing_original_bodies": [
                spec[3] for spec in self.specs if spec not in self.bindings
            ],
            "original_callable_identities_unchanged": identities_stable
            and stock_stable,
            "source_mismatches": dict(self.issues),
            "unfinished_selected_frames": len(self.frames),
            "exceptional_or_phase_crossing_completion_gaps": gaps,
            "monitoring_global_events_zero": global_zero,
            "monitoring_callbacks_owned": callbacks_owned,
            "monitoring_local_masks_owned_before_stop": masks_owned,
            "monitoring_local_masks_cleared_while_active": masks_cleared,
            "monitoring_tool_freed": tool_freed,
            "monitoring_owned_and_restored": owned
            and global_zero
            and masks_owned
            and masks_cleared
            and callbacks_owned
            and tool_freed,
            "profile_owned_and_restored": owned
            and global_zero
            and masks_owned
            and masks_cleared
            and callbacks_owned
            and tool_freed,
            "inactive_callback_events_at_stop": self.callbacks_after_inactive,
            "coverage": "Selected original Python entries and actual normal returned buffers; successful original Windows Native facade HANDLE returns. Distinct API counts overlap, not total kernel/syscall I/O. Starts without returns are explicit gaps, not proven completed refusals.",
            "missing_coverage": "Exceptional completion (local PY_UNWIND unavailable on 3.12), unselected direct open/read APIs, native C SQLite/WAL/SHM/mmap, helper processes, custom streams, direct unobserved native calls, relative HANDLE-parent paths and POSIX syscall totals. Unique paths are observed lexical spellings only.",
        }
