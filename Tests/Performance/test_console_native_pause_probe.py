"""TASK-34402: native, call-through diagnostics, never a timing pass/fail gate.

Real full app, private profile, file-backed DB, controller and capture. Only
the final provider adapter replies immediately. No storage gate is mocked.
Pilot timing includes headless dispatch overhead; per-seam main-thread time
and sampled stacks distinguish blocking app work from that overhead. Nested
seam times overlap and must never be summed across seams.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import json
import os
import platform
import sqlite3
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test


class Observation:
    def __init__(self):
        self.phase = "startup"
        self.main = threading.get_ident()
        self.lock = threading.Lock()
        self.rows = collections.defaultdict(lambda: [0, 0.0, 0.0])
        self.samples = collections.Counter()
        self.slow = []
        self.lags = collections.defaultdict(list)
        self.stages = []
        self.phases = {}
        self.stop = threading.Event()
        self.package = str(Path(__file__).resolve().parents[2] / "tldw_chatbook")

    def stack(self, frame):
        names = []
        for _ in range(200):
            if frame is None:
                break
            code = frame.f_code
            if code.co_filename.startswith(self.package):
                relative = code.co_filename[len(self.package) + 1 :].replace("\\", "/")
                names.append(f"{relative}:{frame.f_lineno}:{code.co_name}")
            frame = frame.f_back
        return " > ".join(reversed(names))

    def record(self, label, phase, elapsed, caller=""):
        thread = "main" if threading.get_ident() == self.main else "worker"
        with self.lock:
            row = self.rows[(phase, thread, label, caller)]
            row[0] += 1
            row[1] += elapsed
            row[2] = max(row[2], elapsed)
            if thread == "main" and elapsed >= 0.1 and len(self.slow) < 100:
                self.slow.append(
                    dict(
                        phase=phase,
                        seam=label,
                        seconds=elapsed,
                        stack=self.stack(sys._getframe(2)),
                    )
                )

    def wrap(self, monkeypatch, owner, name, *, context=False, callers=True):
        original = getattr(owner, name)
        label = f"{getattr(owner, '__name__', type(owner).__name__)}.{name}"

        def enter():
            return (
                self.phase,
                time.perf_counter(),
                self.stack(sys._getframe(2)) if callers else "",
            )

        if context:

            @contextlib.contextmanager
            @functools.wraps(original)
            def measured(*args, **kwargs):
                phase, started, caller = enter()
                manager = original(*args, **kwargs)
                try:
                    active = manager.__enter__()
                finally:
                    self.record(
                        label + ".enter", phase, time.perf_counter() - started, caller
                    )
                try:
                    yield active
                except BaseException:
                    if not manager.__exit__(*sys.exc_info()):
                        raise
                else:
                    manager.__exit__(None, None, None)
        else:

            @functools.wraps(original)
            def measured(*args, **kwargs):
                phase, started, caller = enter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.record(label, phase, time.perf_counter() - started, caller)

        monkeypatch.setattr(owner, name, measured)

    def install(self, monkeypatch):
        from tldw_chatbook.Backup_Recovery import config_participants, storage_admission
        from tldw_chatbook.DB.private_sqlite_process import HelperLease

        self.wrap(monkeypatch, config_participants, "operation", context=True)
        for name in (
            "_acquire_storage",
            "_scope",
            "_local_pause_requested",
            "_observe_candidates",
            "_reuse_evidence",
        ):
            self.wrap(monkeypatch, storage_admission, name)
        self.wrap(
            monkeypatch, storage_admission._Acquisition, "initializing", context=True
        )
        # This wraps the resolved bound method while retaining its classmethod.
        original_start = HelperLease.start

        def start(_cls, *args, **kwargs):
            phase, started = self.phase, time.perf_counter()
            try:
                return original_start(*args, **kwargs)
            finally:
                self.record("HelperLease.start", phase, time.perf_counter() - started)

        monkeypatch.setattr(HelperLease, "start", classmethod(start))
        if os.name == "nt":
            from tldw_chatbook.Utils.windows_files import _Native

            for name in ("open_handle", "security", "ntfs"):
                self.wrap(monkeypatch, _Native, name, callers=False)

        # Audit events count real POSIX os.open without changing its identity
        # (raw participants test membership of os.supports_dir_fd).
        def audit(event, args):
            if not self.stop.is_set() and event == "open" and args[1] is None:
                self.record("os.open.audit", self.phase, 0.0)

        sys.addaudithook(audit)

        def sample():
            while not self.stop.wait(0.05):
                stack = self.stack(sys._current_frames().get(self.main))
                if stack:
                    with self.lock:
                        self.samples[(self.phase, stack)] += 1

        self.sampler = threading.Thread(target=sample, daemon=True)
        self.sampler.start()

    async def heartbeat(self):
        while not self.stop.is_set():
            expected = time.perf_counter() + 0.02
            phase = self.phase
            await asyncio.sleep(0.02)
            self.lags[phase].append(max(0, time.perf_counter() - expected))

    @contextlib.contextmanager
    def phase_scope(self, name):
        self.phase = name
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = time.perf_counter() - started

    def write(self, result):
        requested = os.environ.get("TLDW_PAUSE_PROBE_RESULT")
        if not requested:
            return
        target = Path(requested).resolve()
        root = Path(
            os.environ.get("RUNNER_TEMP") or __import__("tempfile").gettempdir()
        ).resolve()
        assert target.is_relative_to(root), "probe evidence must stay in runner temp"
        with self.lock:
            result.update(
                platform=sys.platform,
                os=platform.platform(),
                python=sys.version,
                sqlite=sqlite3.sqlite_version,
                revision=os.environ.get("GITHUB_SHA", "local"),
                phases_seconds=self.phases,
                stages=self.stages,
                seam_times=[
                    dict(
                        phase=p,
                        thread=t,
                        seam=s,
                        caller=c,
                        count=v[0],
                        seconds=v[1],
                        max_seconds=v[2],
                    )
                    for (p, t, s, c), v in self.rows.items()
                ],
                sampled_stacks=[
                    dict(phase=p, stack=s, samples=n)
                    for (p, s), n in self.samples.most_common(150)
                ],
                slow_main_calls=self.slow,
                heartbeat={
                    p: dict(
                        count=len(v),
                        max_seconds=max(v, default=0),
                        over_100ms=sum(x > 0.1 for x in v),
                    )
                    for p, v in self.lags.items()
                },
                limits="One native run per host. Call-through instrumentation overhead. Pilot/headless dispatch is included in wall time. Nested seam times overlap. Legacy trace maintenance held for send-race isolation; other normal app timers remain enabled. No real provider, OS keyring or model download.",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@private_profile_test
async def test_native_console_pause_probe(monkeypatch, tmp_path, request):
    from Tests.Performance.test_console_keystroke_work_census import _scratch_env
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat import console_runtime, console_send_diagnostics
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunStatus,
    )
    from textual.pilot import Pilot

    _scratch_env(monkeypatch, tmp_path)
    # A known independent revision-GC race can invalidate an admitted revision
    # before call reservation. Match the existing captured-send test isolation.
    monkeypatch.setattr(
        console_runtime, "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS", 3600.0
    )
    original_wait = Pilot._wait_for_screen

    async def wait(self, timeout=180):
        return await original_wait(self, timeout=max(timeout, 180))

    monkeypatch.setattr(Pilot, "_wait_for_screen", wait)
    observed = Observation()
    observed.install(monkeypatch)
    original_stage = console_send_diagnostics.record_send_stage

    def stage(name, status="entered", **kwargs):
        observed.stages.append(
            dict(
                phase=observed.phase,
                stage=name,
                outcome=status,
                time=time.perf_counter(),
            )
        )
        return original_stage(name, status, **kwargs)

    monkeypatch.setattr(console_send_diagnostics, "record_send_stage", stage)
    result = {"complete": False, "provider_calls": 0}
    heartbeat = asyncio.create_task(observed.heartbeat())
    try:
        app = TldwCli()
        async with app.run_test(size=(140, 42)) as pilot:
            screen = app.screen
            assert type(screen).__name__ == "ChatScreen"
            composer = screen._console_composer_or_none()
            assert composer is not None
            controller = screen._ensure_console_chat_controller()
            gateway = controller.provider_gateway
            original_resolve = gateway.resolve_for_send

            async def resolve(selection):
                return replace(await original_resolve(selection), streaming=False)

            def adapter(**_kwargs):
                result["provider_calls"] += 1
                return {
                    "choices": [
                        {"message": {"content": "Immediate native probe reply"}}
                    ]
                }

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)
            monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
            runtime = screen._console_runtime()
            tasks = []
            original_accept = runtime.accept_turn

            def accept(request, **kwargs):
                turn_id = original_accept(request, **kwargs)
                tasks.append(runtime._turn_custody[turn_id].task)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", accept)
            with observed.phase_scope("idle"):
                await asyncio.sleep(3)
            # Direct ticks time only the actual callback, avoiding Pilot cost.
            with observed.phase_scope("credential_poll"):
                for _ in range(8):
                    screen._poll_console_credential_readiness()
            with observed.phase_scope("typing"):
                for _ in range(8):
                    await pilot.press("a")
                await asyncio.sleep(0.5)
            for index in range(1, 4):
                with observed.phase_scope(f"send_{index}"):
                    screen._session._sync_console_session_draft()
                    composer.load_draft(f"Native pause probe message {index}")
                    before = len(tasks)
                    await screen._send_console_message_from_visible_action(
                        session_id=controller.store.active_session_id
                    )
                    assert len(tasks) == before + 1, "send never reached turn custody"
                    await asyncio.wait_for(asyncio.shield(tasks[-1]), 180)
                    assert controller.run_state.status is ConsoleRunStatus.COMPLETED
                    await screen._sync_native_console_chat_ui()
                    await asyncio.sleep(0.25)
            messages = controller.store.messages_for_session(
                controller.store.active_session_id
            )
            result["user_messages"] = sum(
                m.role is ConsoleMessageRole.USER for m in messages
            )
            result["assistant_messages"] = sum(
                m.role is ConsoleMessageRole.ASSISTANT for m in messages
            )
            result["file_backed_database"] = (
                str(app.chachanotes_db.db_path) != ":memory:"
            )
            assert result["file_backed_database"]
            assert (
                result["user_messages"]
                == result["assistant_messages"]
                == result["provider_calls"]
                == 3
            )
            with app.chachanotes_db.transaction() as cursor:
                result["trace_states"] = [
                    row[0]
                    for row in cursor.execute("SELECT state FROM console_trace_calls")
                ]
                result["response_links"] = cursor.execute(
                    "SELECT COUNT(*) FROM console_trace_response_links"
                ).fetchone()[0]
                result["dispatch_checkpoints"] = cursor.execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints"
                ).fetchone()[0]
            assert result["trace_states"] == ["complete"] * 3
            assert result["response_links"] == 3
            assert result["dispatch_checkpoints"] == 0
            result["complete"] = True
            observed.phase = "shutdown"
    finally:
        observed.stop.set()
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        observed.sampler.join(timeout=2)
        observed.write(result)
