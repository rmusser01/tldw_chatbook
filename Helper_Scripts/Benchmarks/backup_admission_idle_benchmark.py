"""Three alternating paired full-app live idle windows, separate from fixed probe.

Requires controller-frozen contemporary baseline/final snapshots using the
fixed probe's source manifest. Optional installed roots/wheels come from the
existing native_package fixture. This script does not build or download them.
No monitor/credential/trace/scheduler timer is parked, driven or stretched.
Receipts contain counters/provenance only; child logs remain private.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import runpy
import shutil
import statistics
import subprocess  # nosec B404 - existing fixed local containment seam
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

FIXED = Path(__file__).with_name("backup_admission_benchmark.py")
spec = importlib.util.spec_from_file_location("fixed_admission_probe", FIXED)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
WINDOW_SECONDS = 60.0
SETTLEMENT = "ui-ready-plus-live-probe-and-five-seconds"
ORDER = (("baseline", "final"), ("final", "baseline"), ("baseline", "final"))
_AUDIT_CENSUS = None
PROCESS_EVENTS = frozenset(
    (
        "subprocess.Popen",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.system",
        "os.exec",
        "os.spawn",
        "pty.spawn",
        "_winapi.CreateProcess",
        "_posixsubprocess.fork_exec",
    )
)


class Census:
    """All-thread call-through counters with atomic cutoff and visible custody."""

    def __init__(self):
        self.lock = threading.RLock()
        self.phase = "settlement"
        units = [
            "os_opens",
            "native_handle_opens",
            "native_acl_reads",
            "storage_admissions",
            "storage_admission_successes",
            "helper_starts",
            "child_starts",
            "probe_starts",
            "probe_completions",
            "probe_errors",
            "probe_overlap",
        ]
        for kind in ("config", "repository"):
            units.extend(
                f"{kind}_{unit}"
                for unit in (
                    "entry_attempts",
                    "entries",
                    "retirement_attempts",
                    "retirements",
                )
            )
        self.counts = {
            phase: dict.fromkeys(units, 0)
            for phase in ("settlement", "window", "drain")
        }
        self.active_scopes = self.active_probes = self.completed_probes = 0
        self.children = []
        self.child_receipts = []
        self.uncovered_process_events = {}
        self.launch_observation = threading.local()

    def audit(self, event, args):
        """Count opens and refuse process routes outside the one observed launch."""
        if event == "open" and args[1] is None:
            self.bump("os_opens")
        elif event in PROCESS_EVENTS and not getattr(
            self.launch_observation, "known", False
        ):
            with self.lock:
                if self.phase is not None:
                    self.uncovered_process_events[event] = (
                        self.uncovered_process_events.get(event, 0) + 1
                    )

    def bump(self, key):
        """Count attempts without changing the wrapped native operation."""
        with self.lock:
            if self.phase is not None:
                self.counts[self.phase][key] += 1

    def install_audit(self):
        """Keep os.open's actual identity, including raw guard membership."""
        global _AUDIT_CENSUS
        if _AUDIT_CENSUS is None:

            def audit(event, args):
                _AUDIT_CENSUS.audit(event, args)

            sys.addaudithook(audit)
        _AUDIT_CENSUS = self

    @contextmanager
    def scope(self, factory, kind):
        """A failed exit retains unresolved custody rather than claiming close."""
        self.bump(f"{kind}_entry_attempts")
        with factory() as value:
            with self.lock:
                self.active_scopes += 1
                self.bump(f"{kind}_entries")
            try:
                yield value
            finally:
                self.bump(f"{kind}_retirement_attempts")
        with self.lock:
            self.active_scopes -= 1
            self.bump(f"{kind}_retirements")

    def cutoff(self):
        """Close the window atomically; delayed work is billed to explicit drain."""
        with self.lock:
            unresolved = {
                "scopes": self.active_scopes,
                "probes": self.active_probes,
                "children": sum(child.poll() is None for child in self.children),
            }
            self.phase = "drain"
            return unresolved

    def instrument(self, stack, source=None):
        """Reuse census seams, native call-through and owned-child observation."""
        from tldw_chatbook.Backup_Recovery import config_participants
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from tldw_chatbook.Backup_Recovery.participants import _RepositoryParticipant
        from tldw_chatbook.DB import private_sqlite_process
        from tldw_chatbook.DB.private_sqlite_process import HelperLease

        for owner, name, kind in (
            (config_participants, "operation", "config"),
            (_RepositoryParticipant, "operation", "repository"),
        ):
            original = getattr(owner, name)
            depth = threading.local()

            @contextmanager
            def operation(
                *args, _original=original, _kind=kind, _depth=depth, **kwargs
            ):
                level = getattr(_depth, "level", 0)
                _depth.level = level + 1
                try:
                    if level == 0:
                        with self.scope(
                            lambda: _original(*args, **kwargs), _kind
                        ) as value:
                            yield value
                    else:
                        with _original(*args, **kwargs) as value:
                            yield value
                finally:
                    _depth.level = level

            stack.enter_context(patch.object(owner, name, operation))

        original_acquire = storage._acquire_storage

        def acquire(*args, **kwargs):
            self.bump("storage_admissions")
            value = original_acquire(*args, **kwargs)
            self.bump("storage_admission_successes")
            return value

        stack.enter_context(patch.object(storage, "_acquire_storage", acquire))
        original_start = HelperLease.start

        def start(cls, *args, **kwargs):
            self.bump("helper_starts")
            previous = getattr(self.launch_observation, "helper", False)
            self.launch_observation.helper = True
            try:
                return original_start(*args, **kwargs)
            finally:
                self.launch_observation.helper = previous

        stack.enter_context(patch.object(HelperLease, "start", classmethod(start)))

        original_spawn = subprocess.Popen.__init__
        helper_entry = (
            Path(private_sqlite_process.__file__)
            .with_name("private_sqlite_helper_entry.py")
            .resolve(strict=True)
        )

        def spawn(child, *args, **kwargs):
            self.bump("child_starts")
            command = args[0] if args else kwargs.get("args")
            observation = None
            if (
                source is not None
                and getattr(self.launch_observation, "helper", False)
                and command
                == [
                    sys.executable,
                    "-I",
                    "-S",
                    str(helper_entry),
                ]
            ):
                receipt = (
                    Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
                    / f"idle-helper-{uuid4().hex}.json"
                )
                digest = hashlib.sha256(helper_entry.read_bytes()).hexdigest()
                observed = [
                    sys.executable,
                    "-I",
                    "-S",
                    str(Path(__file__).resolve()),
                    "--helper-observer",
                    "--source",
                    str(source),
                    "--entry",
                    str(helper_entry),
                    "--entry-sha256",
                    digest,
                    "--receipt",
                    str(receipt),
                ]
                if args:
                    args = (observed, *args[1:])
                else:
                    kwargs["args"] = observed
                observation = {
                    "receipt": receipt,
                    "entry_sha256": digest,
                    "guard_sha256": hashlib.sha256(
                        (source / "Tests/network_guard.py").read_bytes()
                    ).hexdigest(),
                }
            self.launch_observation.known = observation is not None
            try:
                original_spawn(child, *args, **kwargs)
            finally:
                self.launch_observation.known = False
            with self.lock:
                self.children.append(child)
                self.child_receipts.append((child, observation))

        stack.enter_context(patch.object(subprocess.Popen, "__init__", spawn))
        original_probe = storage._local_pause_requested

        def probe(*args, **kwargs):
            with self.lock:
                self.bump("probe_starts")
                if self.active_probes:
                    self.bump("probe_overlap")
                self.active_probes += 1
            try:
                result = original_probe(*args, **kwargs)
            except BaseException:
                self.bump("probe_errors")
                raise
            else:
                self.bump("probe_completions")
                with self.lock:
                    self.completed_probes += 1
                return result
            finally:
                with self.lock:
                    self.active_probes -= 1

        stack.enter_context(patch.object(storage, "_local_pause_requested", probe))
        if os.name == "nt":
            from tldw_chatbook.Utils.windows_files import _Native

            for name, key in (
                ("open_handle", "native_handle_opens"),
                ("security", "native_acl_reads"),
            ):
                original = getattr(_Native, name)

                def native(*args, _original=original, _key=key, **kwargs):
                    self.bump(_key)
                    return _original(*args, **kwargs)

                stack.enter_context(patch.object(_Native, name, native))

    def project_children(self, measured):
        """Join bounded known-helper events to the real monotonic phase cutoffs."""
        measured["parent_window"] = dict(measured.get("window", self.counts["window"]))
        coverage, receipts, network = not self.uncovered_process_events, [], 0
        for child, observation in self.child_receipts:
            if (
                observation is None
                or child.poll() != 0
                or not observation["receipt"].is_file()
            ):
                coverage = False
                continue
            try:
                if observation["receipt"].stat().st_size > 1024**2:
                    raise ValueError("oversized_helper_receipt")
                receipt_bytes = observation["receipt"].read_bytes()
                receipt = json.loads(receipt_bytes)
                if (
                    receipt["schema"] != 1
                    or receipt["pid"] != child.pid
                    or receipt["parent_pid"] != os.getpid()
                    or receipt["entry_sha256"] != observation["entry_sha256"]
                    or receipt["guard_sha256"] != observation["guard_sha256"]
                    or receipt["observer_sha256"]
                    != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    or receipt["exit_code"] != 0
                    or receipt["entry_stable"] is not True
                    or receipt["original_entry_executed"] is not True
                    or receipt["overflow"]
                    or receipt["uncovered_descendants"]
                    or (os.name == "nt" and not receipt["windows_native_observed"])
                ):
                    raise ValueError("incomplete_helper_observation")
                if (
                    not isinstance(receipt["events"], list)
                    or len(receipt["events"]) > 4096
                ):
                    raise ValueError("invalid_helper_event_count")
                for event in receipt["events"]:
                    timestamp, unit = event["timestamp_ns"], event["unit"]
                    if (
                        type(timestamp) is not int
                        or timestamp < 0
                        or unit
                        not in (
                            "os_opens",
                            "native_handle_opens",
                            "native_acl_reads",
                            "network_attempts",
                        )
                    ):
                        raise ValueError("invalid_helper_event")
                    if unit == "network_attempts":
                        network += 1
                    elif "window_started_ns" in measured:
                        phase = (
                            "settlement"
                            if timestamp < measured["window_started_ns"]
                            else "window"
                            if timestamp < measured["cutoff_ns"]
                            else "drain"
                        )
                        target = (
                            measured["window"]
                            if phase == "window"
                            else self.counts[phase]
                        )
                        target[unit] += 1
                receipts.append(
                    {
                        "pid": child.pid,
                        "receipt": str(observation["receipt"]),
                        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                        "entry_sha256": receipt["entry_sha256"],
                        "guard_sha256": receipt["guard_sha256"],
                        "events": len(receipt["events"]),
                        "retired": True,
                    }
                )
            except (KeyError, TypeError, ValueError, OSError):
                coverage = False
        return {
            "child_network_coverage": "joined"
            if coverage and self.children
            else "no-descendants"
            if not self.children and coverage
            else "unresolved",
            "child_receipts": receipts,
            "child_network_attempts": network,
            "uncovered_process_events": dict(self.uncovered_process_events),
        }


def helper_observer(source, entry, expected_digest, receipt_path):
    """Observe one exact -I/-S entry; its original main owns namespaces/predicates."""
    events, lock = [], threading.Lock()
    result = {
        "schema": 1,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "exit_code": 1,
        "entry_sha256": expected_digest,
        "entry_stable": False,
        "original_entry_executed": False,
        "overflow": False,
        "uncovered_descendants": False,
        "windows_native_observed": False,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "events": events,
    }

    def event(unit):
        with lock:
            if len(events) == 4096:
                result["overflow"] = True
            else:
                events.append({"timestamp_ns": time.monotonic_ns(), "unit": unit})

    def audit(name, args):
        if name == "open" and args[1] is None:
            event("os_opens")
        elif name in PROCESS_EVENTS:
            result["uncovered_descendants"] = True

    try:
        if (
            not sys.flags.isolated
            or not sys.flags.no_site
            or hashlib.sha256(entry.read_bytes()).hexdigest() != expected_digest
        ):
            raise ValueError("helper_entry_identity_changed")
        sys.addaudithook(audit)
        guard_path = source / "Tests/network_guard.py"
        guard_spec = importlib.util.spec_from_file_location(
            "Tests.network_guard", guard_path
        )
        guard = importlib.util.module_from_spec(guard_spec)
        sys.modules[guard_spec.name] = guard
        guard_spec.loader.exec_module(guard)
        result["guard_sha256"] = hashlib.sha256(guard_path.read_bytes()).hexdigest()
        original_deny = guard._deny

        def deny(*args, **kwargs):
            event("network_attempts")
            return original_deny(*args, **kwargs)

        guard._deny = deny
        guard.install()
        guard.set_allowed(False)
        if os.name == "nt":

            def imported(frame, name, arg):
                if (
                    name == "return"
                    and frame.f_code.co_name == "<module>"
                    and frame.f_globals.get("__name__")
                    == "tldw_chatbook.Utils.windows_files"
                ):
                    native_type = frame.f_globals["_Native"]
                    for method, unit in (
                        ("open_handle", "native_handle_opens"),
                        ("security", "native_acl_reads"),
                    ):
                        original = getattr(native_type, method)

                        def native(*args, _original=original, _unit=unit, **kwargs):
                            event(_unit)
                            return _original(*args, **kwargs)

                        setattr(native_type, method, native)
                    result["windows_native_observed"] = True
                    sys.setprofile(None)

            sys.setprofile(imported)
        sys.argv = [str(entry)]
        result["original_entry_executed"] = True
        try:
            runpy.run_path(str(entry), run_name="__main__")
            result["exit_code"] = 0
        except SystemExit as error:
            result["exit_code"] = (
                error.code
                if type(error.code) is int
                else 0
                if error.code is None
                else 1
            )
        result["entry_stable"] = (
            hashlib.sha256(entry.read_bytes()).hexdigest() == expected_digest
        )
    except BaseException as error:  # noqa: BLE001 - no private child exception text
        result["error_type"] = type(error).__name__[:80]
    finally:
        sys.setprofile(None)
        receipt_path.write_text(json.dumps(result, sort_keys=True) + "\n")
        receipt_path.chmod(0o600)
    return result["exit_code"]


def private_environment(profile):
    """Reuse fixed private profile and replace temp/cache/state ambient paths."""
    environment = fixed.private_environment(profile)
    for key, leaf in (
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_STATE_HOME", "state"),
        ("TEMP", "tmp"),
        ("TMP", "tmp"),
        ("TMPDIR", "tmp"),
    ):
        directory = profile / leaf
        directory.mkdir(mode=0o700, exist_ok=True)
        environment[key] = str(directory)
    return environment


def package_files(root):
    """Hash production Python/SQL bytes/stat identities without following links."""
    result = {}
    for path in sorted((root / "tldw_chatbook").rglob("*")):
        if path.suffix not in (".py", ".sql"):
            continue
        if path.is_symlink():
            raise ValueError("linked_production_module")
        info = path.stat()
        result[path.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stat": [
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ],
        }
    if not result:
        raise ValueError("missing_production_modules")
    return result


def installed_join(source, installed, wheel):
    """Join every installed production Python byte to source and existing wheel."""
    source_files, installed_files = package_files(source), package_files(installed)
    if source_files.keys() != installed_files.keys():
        raise ValueError("installed_source_wheel_mismatch")
    with zipfile.ZipFile(wheel) as archive:
        for name, info in installed_files.items():
            if (
                info["sha256"] != source_files[name]["sha256"]
                or hashlib.sha256(archive.read(name)).hexdigest() != info["sha256"]
            ):
                raise ValueError("installed_source_wheel_mismatch")
    return {
        "joined": True,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "installed_files": installed_files,
    }


def producer_state(app):
    """Read actual supported Task/Timer/Worker state; never drive a producer."""
    from textual.timer import Timer
    from textual.worker import Worker

    def task_state(task):
        known = isinstance(task, asyncio.Task)
        return {
            "identity": id(task) if task is not None else None,
            "known": known,
            "running": known
            and not task.done()
            and not task.cancelled()
            and task.cancelling() == 0,
        }

    screen = getattr(app, "screen", None)
    runtime = getattr(screen, "_console_runtime_ref", None)
    timer = getattr(screen, "_console_credential_poll_timer", None)
    worker = getattr(app, "scheduler_worker", None)
    active = getattr(timer, "_active", None)
    known_timer = isinstance(timer, Timer) and isinstance(active, asyncio.Event)
    known_worker = isinstance(worker, Worker)
    return {
        "screen_identity": id(screen) if screen is not None else None,
        "runtime_identity": id(runtime) if runtime is not None else None,
        "maintenance_error_clear": hasattr(app, "_backup_maintenance_error")
        and app._backup_maintenance_error is None,
        "monitor": task_state(getattr(app, "_backup_maintenance_monitor_task", None)),
        "trace": task_state(getattr(runtime, "_legacy_trace_maintenance_task", None)),
        "credential_timer": {
            "identity": id(timer) if timer is not None else None,
            "known": known_timer,
            "active": known_timer and active.is_set(),
            "interval": timer._interval if known_timer else None,
            "repeat": timer._repeat if known_timer else None,
            "task": task_state(getattr(timer, "_task", None)),
        },
        "scheduler": {
            "identity": id(worker) if worker is not None else None,
            "known": known_worker,
            "running": known_worker and worker.is_running,
            "cancelled": not known_worker or worker.is_cancelled,
            "finished": not known_worker or worker.is_finished,
            "error_clear": known_worker and worker.error is None,
            "task": task_state(getattr(worker, "_task", None)),
        },
    }


def producers_live(state):
    """Unknown, paused, cancelling, failed or stopped producers cannot qualify."""
    timer, worker = state["credential_timer"], state["scheduler"]
    return bool(
        state["screen_identity"] is not None
        and state["runtime_identity"] is not None
        and state["maintenance_error_clear"] is True
        and all(
            state[name]["known"] is True and state[name]["running"] is True
            for name in ("monitor", "trace")
        )
        and timer["known"] is True
        and timer["active"] is True
        and timer["task"]["known"] is True
        and timer["task"]["running"] is True
        and worker["known"] is True
        and worker["running"] is True
        and worker["cancelled"] is False
        and worker["finished"] is False
        and worker["error_clear"] is True
        and worker["task"]["known"] is True
        and worker["task"]["running"] is True
    )


async def live_window(census, app=None):
    """One monotonic real 60-second wait, no pilot stimulus or timer changes."""
    before = producer_state(app) if app is not None else None
    if before is not None and not producers_live(before):
        raise RuntimeError("normal_idle_producers_unavailable")
    with census.lock:
        census.phase = "window"
        started = time.monotonic_ns()
    await asyncio.sleep(WINDOW_SECONDS)
    with census.lock:
        cutoff_state = producer_state(app) if app is not None else None
        elapsed = time.monotonic_ns() - started
        unresolved = census.cutoff()
        window = dict(census.counts["window"])
    result = {
        "elapsed_ns": elapsed,
        "window_started_ns": started,
        "cutoff_ns": started + elapsed,
        "window_seconds": WINDOW_SECONDS,
        "window": window,
        "unresolved_at_cutoff": unresolved,
    }
    if before is not None:
        result.update(
            producer_checks={"before": before, "cutoff": cutoff_state},
            timers_live=before == cutoff_state and producers_live(cutoff_state),
        )
    return result


async def idle(census):
    """Actual readiness, live settlement, normal idle, then public app teardown."""
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    measured = {}
    async with app.run_test(size=(120, 40)):
        async with asyncio.timeout(60 if os.name == "nt" else 20):
            while not app._ui_ready:
                await asyncio.sleep(0.005)
        async with asyncio.timeout(30):
            while not census.completed_probes or census.active_probes:
                await asyncio.sleep(0.005)
            await asyncio.sleep(5)
        measured = await live_window(census, app)
        measured.update(ui_ready=True, settled=True, settlement=SETTLEMENT)
    return measured


def child(source, installed=None, wheel=None, seed=False):
    """Guard before imports; exact bytes before/after, and positive retirement."""
    result = {
        "exit_code": 1,
        "retired": False,
        "source_stable": False,
        "installed_stable": False,
        "protocol": "live-idle-v1",
        "platform": sys.platform,
        "execution_mode": "installed" if installed else "source",
    }
    census, guard = Census(), None
    source = source.resolve(strict=True)
    installed = installed.resolve(strict=True) if installed else None
    selected = installed or source
    started_drain = None
    try:
        result["source"] = fixed.select_source(source)
        before = package_files(selected)
        result["installed_join"] = (
            installed_join(source, installed, wheel) if installed else None
        )
        guard = fixed.prepare_imports(source)
        if installed:
            sys.path.insert(0, str(installed))
        census.install_audit()
        with ExitStack() as stack:
            census.instrument(stack, source)
            if seed:
                result.update(fixed.transaction(fixed.counters(), 1, seed=True))
            else:
                result.update(asyncio.run(idle(census)))
            started_drain = result.get("cutoff_ns", time.monotonic_ns())
            census.phase = "drain"
            result.update(fixed.retired_state(census.children))
            result["exit_code"] = 0
        result["source_stable"] = fixed.select_source(source) == result["source"]
        result["installed_stable"] = package_files(selected) == before
        result["production_files"] = before
    except BaseException as error:  # noqa: BLE001 - content-free failure projection
        result["error_type"] = type(error).__name__[:80]
    finally:
        census.phase = "drain"
        if (
            not result["retired"]
            and "tldw_chatbook.Backup_Recovery.storage_admission" in sys.modules
        ):
            try:
                result.update(fixed.retired_state(census.children))
            except BaseException as error:  # noqa: BLE001 - failed cleanup stays failed
                result["cleanup_error_type"] = type(error).__name__[:80]
        census.phase = None
        result["drain_ns"] = (
            time.monotonic_ns() - started_drain if started_drain else None
        )
        result["drain"] = dict(census.counts["drain"])
        result.setdefault("window", dict(census.counts["window"]))
        result["settlement_counts"] = dict(census.counts["settlement"])
        result["unresolved_after_cleanup"] = {
            "scopes": census.active_scopes,
            "probes": census.active_probes,
        }
        result["parent_network_attempts"] = (
            len(guard.blocked_attempts()) if guard else 0
        )
        result.update(census.project_children(result))
        result["network_attempts"] = (
            result["parent_network_attempts"] + result["child_network_attempts"]
        )
        result["drain"] = dict(census.counts["drain"])
        result["settlement_counts"] = dict(census.counts["settlement"])
        result["observed_children"] = [
            {"pid": child.pid, "returncode": child.poll()} for child in census.children
        ]
        result["imported_modules"] = [
            {
                "module": name,
                "file": str(Path(module.__file__).resolve()),
                "sha256": hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest(),
            }
            for name, module in tuple(sys.modules.items())
            if name.startswith("tldw_chatbook")
            and getattr(module, "__file__", None)
            and Path(module.__file__).is_file()
        ]
        result["foreign_source_modules"] = sum(
            not Path(module.__file__).resolve().is_relative_to(selected)
            for name, module in tuple(sys.modules.items())
            if name.startswith("tldw_chatbook") and getattr(module, "__file__", None)
        )
        if (
            not result["retired"]
            or not result["source_stable"]
            or not result["installed_stable"]
            or result["network_attempts"]
            or result["foreign_source_modules"]
            or any(result["unresolved_after_cleanup"].values())
            or (not seed and result.get("timers_live") is not True)
            or result["child_network_coverage"] == "unresolved"
        ):
            result["exit_code"] = 1
    return result


def run_child(source, profile, label, installed=None, wheel=None, seed=False):
    """Unique retained receipts and existing 300s owned-tree supervision."""
    if shutil.disk_usage(profile.parent.parent).free < 2 * 1024**3:
        raise RuntimeError("idle_phase_reserve_below_two_gib")
    environment = private_environment(profile)
    environment["TLDW_PROBE_SOURCE"] = str(source)
    result_file = profile / f"{label}-child.json"
    if result_file.exists():
        raise ValueError("idle_receipt_already_exists")
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--child",
        "--source",
        str(source),
        "--receipt",
        str(result_file),
    ]
    if installed:
        command.extend(("--installed", str(installed), "--wheel", str(wheel)))
    if seed:
        command.append("--seed")
    supervised = asyncio.run(
        fixed.supervise(command, source, environment, profile / f"{label}-child.log")
    )
    result = (
        json.loads(result_file.read_text())
        if result_file.is_file()
        else {"exit_code": 1, "retired": False, "error_type": "MissingReceipt"}
    )
    child_status = result["exit_code"]
    result.update(supervised)
    result["exit_code"] = child_status or supervised["exit_code"]
    if not supervised["supervisor_retired"] or supervised.get("supervisor_error_type"):
        result["exit_code"] = result["exit_code"] or 1
    return result


def compare(runs):
    """Fail closed on incomplete, incompatible or unresolved paired windows."""
    result = {"qualified": False, "baseline_rates": [], "final_rates": []}
    try:
        if len(runs) != 6:
            raise ValueError("six_windows_required")
        identities = {}
        matching = (
            "platform",
            "execution_mode",
            "probe_sha256",
            "containment_sha256",
            "dependency_sha256",
            "profile_depth",
            "seed_notes",
            "settlement",
            "window_seconds",
        )
        for index, run in enumerate(runs):
            if run["pair"] != index // 2 or run["side"] != ORDER[index // 2][index % 2]:
                raise ValueError("alternating_pair_order_required")
            if run["protocol"] != "live-idle-v1" or run["platform"] not in (
                "darwin",
                "linux",
                "win32",
            ):
                raise ValueError("unsupported_protocol_or_platform")
            if run["execution_mode"] not in ("source", "installed"):
                raise ValueError("unsupported_execution_mode")
            for key in (
                "ui_ready",
                "settled",
                "timers_live",
                "retired",
                "supervisor_retired",
                "source_stable",
                "installed_stable",
            ):
                if run[key] is not True:
                    raise ValueError("incomplete_lifetime_or_readiness")
            if any(
                run[key] != 0
                for key in ("exit_code", "network_attempts", "foreign_source_modules")
            ):
                raise ValueError("failed_window")
            if (
                run["window_seconds"] != 60.0
                or type(run["elapsed_ns"]) is not int
                or run["elapsed_ns"] < 60_000_000_000
                or not math.isfinite(run["elapsed_ns"])
                or run["settlement"] != SETTLEMENT
                or run["seed_notes"] != 8
            ):
                raise ValueError("window_protocol_changed")
            if (
                set(run["unresolved_at_cutoff"]) != {"scopes", "probes", "children"}
                or any(run["unresolved_at_cutoff"].values())
                or not run["outstanding_ownership"]
                or any(run["outstanding_ownership"].values())
                or run["child_network_coverage"] not in ("no-descendants", "joined")
            ):
                raise ValueError("unresolved_work")
            checks = run["producer_checks"]
            if (
                set(checks) != {"before", "cutoff"}
                or checks["before"] != checks["cutoff"]
                or not producers_live(checks["cutoff"])
                or run["uncovered_process_events"]
            ):
                raise ValueError(
                    "changed_or_unknown_normal_producers_or_process_coverage"
                )
            first_timer = runs[0]["producer_checks"]["before"]["credential_timer"]
            if any(
                checks["before"]["credential_timer"][key] != first_timer[key]
                for key in ("interval", "repeat")
            ):
                raise ValueError("incompatible_credential_timer_cadence")
            counts = run["window"]
            for key in Census().counts["window"]:
                if type(counts[key]) is not int or counts[key] < 0:
                    raise ValueError("invalid_counter")
            for kind in ("config", "repository"):
                if (
                    len(
                        {
                            counts[f"{kind}_{unit}"]
                            for unit in (
                                "entry_attempts",
                                "entries",
                                "retirement_attempts",
                                "retirements",
                            )
                        }
                    )
                    != 1
                ):
                    raise ValueError("incomplete_scope_boundary")
            if counts["storage_admissions"] != counts["storage_admission_successes"]:
                raise ValueError("failed_storage_admission")
            if (
                counts["probe_errors"]
                or counts["probe_overlap"]
                or counts["probe_completions"] == 0
                or counts["probe_starts"] != counts["probe_completions"]
            ):
                raise ValueError("unresolved_or_failed_native_probes")
            if run["platform"] == "win32" and (
                counts["native_handle_opens"] == 0 or counts["native_acl_reads"] == 0
            ):
                raise ValueError("missing_native_windows_costs")
            if any(run[key] != runs[0][key] for key in matching):
                raise ValueError("incompatible_windows")
            source = run["source"]
            if set(source) != {"commit", "tree", "content_sha256"} or any(
                not isinstance(value, str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}" if key == "content_sha256" else r"[0-9a-f]{40}",
                    value,
                )
                for key, value in source.items()
            ):
                raise ValueError("invalid_source_identity")
            if run["side"] in identities and source != identities[run["side"]]:
                raise ValueError("source_identity_changed_between_windows")
            identities[run["side"]] = source
            result[run["side"] + "_rates"].append(
                counts["os_opens"] / (run["elapsed_ns"] / 1e9)
            )
        if identities["baseline"] == identities["final"]:
            raise ValueError("same_source_denominator")
        baseline = statistics.median(result["baseline_rates"])
        final = statistics.median(result["final_rates"])
        if baseline <= 0:
            raise ValueError("zero_baseline_denominator")
        result.update(
            baseline_median_opens_per_second=baseline,
            final_median_opens_per_second=final,
            reduction_percent=100 * (1 - final / baseline),
            baseline_range=[
                min(result["baseline_rates"]),
                max(result["baseline_rates"]),
            ],
            final_range=[min(result["final_rates"]), max(result["final_rates"])],
            qualified=final <= baseline * 0.5,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        result["failure_type"] = type(error).__name__
    return result


def main():
    """Controller supplies contemporary exact source and optional installed pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-description")
    parser.add_argument("--installed", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--baseline-installed", type=Path)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--helper-observer", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--entry", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--entry-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--seed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.helper_observer:
        if not args.entry or not args.entry_sha256:
            parser.error("known_helper_entry_required")
        return helper_observer(args.source, args.entry, args.entry_sha256, args.receipt)
    if bool(args.installed) != bool(args.wheel) or (args.seed and not args.child):
        parser.error("installed_pair_or_seed_invalid")
    if args.child:
        result = child(args.source, args.installed, args.wheel, args.seed)
    else:
        if not args.baseline or not args.baseline_description or args.receipt.exists():
            parser.error("disclosed_baseline_and_fresh_receipt_required")
        if bool(args.installed) != bool(args.baseline_installed) or bool(
            args.baseline_installed
        ) != bool(args.baseline_wheel):
            parser.error("matched_installed_pairs_required")
        sources = {
            "baseline": args.baseline.resolve(strict=True),
            "final": args.source.resolve(strict=True),
        }
        for source in sources.values():
            fixed.select_source(source)
        args.receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="admission-idle-", dir=args.receipt.parent))
        dependency = hashlib.sha256(
            json.dumps(
                sorted(
                    (d.metadata["Name"], d.version)
                    for d in importlib.metadata.distributions()
                )
            ).encode()
        ).hexdigest()
        runs, seeds = [], []
        for pair, order in enumerate(ORDER):
            for side in order:
                profile = root / f"pair-{pair}-{side}" / "profile"
                installed = (
                    args.baseline_installed if side == "baseline" else args.installed
                )
                wheel = args.baseline_wheel if side == "baseline" else args.wheel
                seed = run_child(
                    sources[side],
                    profile,
                    f"{pair}-{side}-seed",
                    installed,
                    wheel,
                    True,
                )
                seeds.append(seed)
                run = (
                    run_child(
                        sources[side], profile, f"{pair}-{side}-idle", installed, wheel
                    )
                    if seed["exit_code"] == 0
                    else {"exit_code": 1, "error_type": "SeedFailed"}
                )
                run.update(
                    side=side,
                    pair=pair,
                    profile_depth=len(profile.parts),
                    seed_notes=8,
                    dependency_sha256=dependency,
                    probe_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    containment_sha256=hashlib.sha256(
                        fixed.CONTAINMENT.read_bytes()
                    ).hexdigest(),
                )
                runs.append(run)
        comparison = compare(runs)
        result = {
            "protocol": "live-idle-v1",
            "baseline_description": args.baseline_description,
            "sources": {
                side: fixed.select_source(source) for side, source in sources.items()
            },
            "seeds": seeds,
            "runs": runs,
            "comparison": comparison,
            "exit_code": int(not comparison["qualified"]),
            "limits": {
                "window_seconds": 60,
                "pairs": 3,
                "reduction_percent": 50,
                "child_timeout_seconds": 300,
            },
        }
    args.receipt.write_text(json.dumps(result, sort_keys=True) + "\n")
    args.receipt.chmod(0o600)
    if not args.child:
        print(
            json.dumps(
                {"exit_code": result["exit_code"], "comparison": result["comparison"]}
            )
        )
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
