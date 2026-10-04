"""Native captured-send regression budgets with call-through pause diagnostics.

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
import hashlib
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


# Recorded before the final integration measurement (TASK-34403). Earlier native
# Windows sends took 53/51/41s with 8-9s UI stalls and >100,000 opens per send;
# Linux/macOS sends were ~3-4s. These bound actual work, never provider latency.
MAX_CAPTURED_SEND_SECONDS = 15
MAX_SEND_UI_STALL_SECONDS = 1
MAX_MAIN_SEND_ADMISSIONS = 200
MAX_WINDOWS_SEND_NATIVE_OPENS = 40_000
MAX_POSIX_SEND_HELPERS = 16


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
        self.config_counts = collections.Counter()
        self.config_details = []
        self.config_detail_phases = collections.Counter()
        self.config_local = threading.local()
        self.phase_windows = {}
        self.delay_windows = []
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

    @staticmethod
    def _metadata_hash(value):
        return (
            None
            if value is None
            else hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
        )

    @staticmethod
    def _posture_differences(before, after):
        differences = []
        before, after = before or (), after or ()
        for index in range(min(max(len(before), len(after)), 64)):
            left = before[index] if index < len(before) else None
            right = after[index] if index < len(after) else None
            if left == right:
                continue
            fields = []
            if isinstance(left, tuple) and isinstance(right, tuple):
                fields = [
                    field
                    for field in range(min(max(len(left), len(right)), 16))
                    if (left[field] if field < len(left) else None)
                    != (right[field] if field < len(right) else None)
                ]
            differences.append(dict(component=index, fields=fields))
        return differences

    def config_record(self, seam, outcome, phase, metadata, caller):
        thread = "main" if threading.get_ident() == self.main else "worker"
        with self.lock:
            self.config_counts[(phase, thread, seam, outcome)] += 1
            if (
                outcome != "hit"
                and len(self.config_details) < 96
                and self.config_detail_phases[phase] < 12
            ):
                self.config_detail_phases[phase] += 1
                self.config_details.append(
                    dict(
                        phase=phase,
                        thread=thread,
                        seam=seam,
                        outcome=outcome,
                        caller=caller,
                        **metadata,
                    )
                )

    def install_config(self, monkeypatch, config):
        """Call through existing checks; observe metadata without additional IO."""
        original_posture = config._config_file_posture
        original_hit = config._settings_cache_hit

        @functools.wraps(original_posture)
        def posture(path):
            phase, started = self.phase, time.perf_counter()
            try:
                result = original_posture(path)
                current = getattr(self.config_local, "cache_probe", None)
                if current is not None:
                    current["observed"] = result
                return result
            finally:
                self.record(
                    "tldw_chatbook.config._config_file_posture",
                    phase,
                    time.perf_counter() - started,
                )

        @functools.wraps(original_hit)
        def cache_hit(path):
            phase, started = self.phase, time.perf_counter()
            before = (
                id(config._SETTINGS_CACHE),
                config._SETTINGS_CACHE_SOURCE,
                config._SETTINGS_CACHE_POSTURE,
            )
            cached_missing = config._SETTINGS_CACHE is None
            previous = getattr(self.config_local, "cache_probe", None)
            current = {}
            self.config_local.cache_probe = current
            try:
                result = original_hit(path)
            finally:
                self.config_local.cache_probe = previous
                self.record(
                    "tldw_chatbook.config._settings_cache_hit",
                    phase,
                    time.perf_counter() - started,
                )
            outcome = (
                "hit"
                if result is not None
                else (
                    "empty"
                    if cached_missing
                    else "source"
                    if before[1] != path
                    else "posture"
                    if "observed" in current and current["observed"] != before[2]
                    else "unobserved_or_raced"
                )
            )
            metadata = {}
            caller = ""
            if outcome != "hit":
                after = (
                    id(config._SETTINGS_CACHE),
                    config._SETTINGS_CACHE_SOURCE,
                    config._SETTINGS_CACHE_POSTURE,
                )
                metadata = dict(
                    source_sha256=self._metadata_hash(before[1]),
                    selected_sha256=self._metadata_hash(path),
                    expected_posture_sha256=self._metadata_hash(before[2]),
                    observed_posture_sha256=self._metadata_hash(
                        current.get("observed")
                    ),
                    changed_fields=self._posture_differences(
                        before[2], current.get("observed")
                    )
                    if "observed" in current
                    else [],
                    cache_state_changed=before != after,
                )
                caller = self.stack(sys._getframe(1))
                self.config_local.last_miss = (phase, outcome)
            self.config_record("_settings_cache_hit", outcome, phase, metadata, caller)
            return result

        monkeypatch.setattr(config, "_config_file_posture", posture)
        monkeypatch.setattr(config, "_settings_cache_hit", cache_hit)
        # Guarded loader identities are part of the installed-source contract.
        # Public-loader counts cover module calls, not previously imported aliases.
        for name in (
            "load_settings",
            "_invalidate_config_caches",
            "set_encryption_password",
            "_set_session_encryption_password",
        ):
            original = getattr(config, name)

            def install_one(name, original):
                @functools.wraps(original)
                def measured(*args, **kwargs):
                    phase, started = self.phase, time.perf_counter()
                    caller = self.stack(sys._getframe(1))
                    forced = bool(
                        kwargs.get(
                            "force_reload",
                            args[0] if args and name == "load_settings" else False,
                        )
                    )
                    outcome = (
                        "forced"
                        if forced
                        else "read"
                        if name == "load_settings"
                        else "invalidate"
                    )
                    recent = getattr(self.config_local, "last_miss", None)
                    metadata = dict(
                        force_reload=forced,
                        preceding_miss=recent[1]
                        if recent and recent[0] == phase
                        else None,
                    )
                    self.config_record(name, outcome, phase, metadata, caller)
                    try:
                        return original(*args, **kwargs)
                    finally:
                        self.record(
                            "tldw_chatbook.config." + name,
                            phase,
                            time.perf_counter() - started,
                            caller,
                        )

                monkeypatch.setattr(config, name, measured)

            install_one(name, original)

    def install(self, monkeypatch):
        from tldw_chatbook import config
        from tldw_chatbook.Backup_Recovery import config_participants, storage_admission
        from tldw_chatbook.DB.private_sqlite_process import HelperLease

        self.install_config(monkeypatch, config)

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
            caller = self.stack(sys._getframe(1))
            try:
                return original_start(*args, **kwargs)
            finally:
                self.record(
                    "HelperLease.start", phase, time.perf_counter() - started, caller
                )

        monkeypatch.setattr(HelperLease, "start", classmethod(start))
        if os.name == "nt":
            from tldw_chatbook.Utils.windows_files import _Native

            for name in ("open_handle", "security", "ntfs", "_token_sid"):
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
            resumed = time.perf_counter()
            self.delay_windows.append((expected, resumed))
            self.lags[phase].append(max(0, resumed - expected))

    @contextlib.contextmanager
    def phase_scope(self, name):
        self.phase = name
        started = time.perf_counter()
        try:
            yield
        finally:
            ended = time.perf_counter()
            self.phases[name] = ended - started
            self.phase_windows[name] = (started, ended)

    def phase_lags(self, phase):
        """Bill delayed heartbeat intervals to every phase window they overlap."""
        if phase not in self.phase_windows:
            return self.lags[phase]
        started, ended = self.phase_windows[phase]
        return [
            max(0, min(resumed, ended) - max(expected, started))
            for expected, resumed in self.delay_windows
            if resumed >= started and expected <= ended
        ]

    def assert_send_budgets(self):
        for phase in ("send_1", "send_2", "send_3"):
            assert self.phases[phase] <= MAX_CAPTURED_SEND_SECONDS, (
                phase,
                "captured send duration",
                self.phases[phase],
            )
            lag = max(self.phase_lags(phase), default=0)
            assert lag <= MAX_SEND_UI_STALL_SECONDS, (phase, "UI stall", lag)
            admissions = sum(
                row[0]
                for (p, thread, seam, _), row in self.rows.items()
                if p == phase
                and thread == "main"
                and seam.endswith("._acquire_storage")
            )
            assert admissions <= MAX_MAIN_SEND_ADMISSIONS, (
                phase,
                "UI storage acquisitions",
                admissions,
            )
            if os.name == "nt":
                native_opens = sum(
                    row[0]
                    for (p, _, seam, _), row in self.rows.items()
                    if p == phase and seam == "_Native.open_handle"
                )
                assert native_opens <= MAX_WINDOWS_SEND_NATIVE_OPENS, (
                    phase,
                    "native file opens",
                    native_opens,
                )
            else:
                helpers = sum(
                    row[0]
                    for (p, _, seam, _), row in self.rows.items()
                    if p == phase and seam == "HelperLease.start"
                )
                assert helpers <= MAX_POSIX_SEND_HELPERS, (
                    phase,
                    "private SQLite helper starts",
                    helpers,
                )

    def write(self, result):
        requested = os.environ.get("TLDW_PAUSE_PROBE_RESULT")
        if not requested:
            return
        target = Path(requested).resolve()
        root = Path(
            os.environ.get("RUNNER_TEMP") or __import__("tempfile").gettempdir()
        ).resolve()
        assert target.is_relative_to(root), "probe evidence must stay in runner temp"
        source_root = Path(__file__).resolve().parents[2]
        loaded_files = {}
        for name, module in tuple(sys.modules.items()):
            if name != "tldw_chatbook" and not name.startswith("tldw_chatbook."):
                continue
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            path = Path(filename).resolve()
            assert path.is_relative_to(source_root), (
                "foreign loaded app source",
                name,
                str(path),
            )
            relative = path.relative_to(source_root).as_posix()
            loaded_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.lock:
            result.update(
                source_root=str(source_root),
                loaded_source_sha256=loaded_files,
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
                config_cache_outcomes=[
                    dict(phase=p, thread=t, seam=s, outcome=o, count=n)
                    for (p, t, s, o), n in self.config_counts.items()
                ],
                config_cache_miss_details=self.config_details,
                heartbeat={
                    p: dict(
                        count=len(v),
                        max_seconds=max(v, default=0),
                        over_100ms=sum(x > 0.1 for x in v),
                    )
                    for p in self.phases.keys() | self.lags.keys()
                    for v in (self.phase_lags(p),)
                },
                limits="One native run per host. Call-through instrumentation overhead. Pilot/headless dispatch is included in wall time. Nested seam times overlap. Normal app timers and startup trace maintenance remain enabled. Heartbeat delay intervals are intersected with phase windows, including synchronous entry stalls. Public config-loader metrics cover module-attribute calls; imported aliases may not be observed. Guarded config callable identities remain installed. No real provider, OS keyring or model download.",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@private_profile_test
async def test_native_console_pause_probe(monkeypatch, tmp_path, request):
    from Tests.Performance.test_console_keystroke_work_census import _scratch_env
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat import console_send_diagnostics
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunStatus,
    )
    from textual.pilot import Pilot

    _scratch_env(monkeypatch, tmp_path)
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
                return replace(
                    await original_resolve(selection), streaming=len(tasks) >= 3
                )

            def adapter(**_kwargs):
                result["provider_calls"] += 1
                streaming = bool(_kwargs.get("streaming"))
                result.setdefault("provider_streaming", []).append(streaming)
                if streaming:
                    common = {
                        "id": "native-pause",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "gpt-4o",
                    }
                    events = [
                        {
                            **common,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "Immediate native probe reply",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            **common,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        },
                    ]
                    return iter(
                        [
                            *(
                                "data: " + json.dumps(event) + "\n\n"
                                for event in events
                            ),
                            "data: [DONE]\n\n",
                        ]
                    )
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
            assert result["provider_streaming"] == [False, False, True]
            # The unchanged real bar must now avoid the previously sampled
            # stylesheet cascade. Retain the same eight-call diagnostic control
            # for before/after comparison; dedicated mounted tests verify paint.
            from tldw_chatbook.Widgets.Console.console_control_bar import (
                ConsoleControlBar,
            )

            bar = screen.query_one(ConsoleControlBar)
            bar._set_recovery_height(False)

            def layout_state():
                return (
                    bar.classes,
                    bar.styles.height,
                    bar.styles.min_height,
                    bar.styles.max_height,
                )

            expected_layout = layout_state()
            with observed.phase_scope("unchanged_layout"):
                for _ in range(8):
                    bar._set_recovery_height(False)
            assert layout_state() == expected_layout
            with observed.phase_scope("unchanged_layout_control"):
                for _ in range(8):
                    if layout_state() != expected_layout:
                        bar._set_recovery_height(False)
            assert layout_state() == expected_layout
            observed.assert_send_budgets()
            result["complete"] = True
            observed.phase = "shutdown"
    finally:
        observed.stop.set()
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        observed.sampler.join(timeout=2)
        observed.write(result)


@pytest.mark.asyncio
async def test_heartbeat_budget_includes_synchronous_send_entry_stall(monkeypatch):
    """A heartbeat begun while typing must still bill the following send stall."""
    monkeypatch.setattr(sys.modules[__name__], "MAX_SEND_UI_STALL_SECONDS", 0.06)
    observed = Observation()
    observed.phases.update(send_2=0.0, send_3=0.0)
    observed.phase = "typing"
    heartbeat = asyncio.create_task(observed.heartbeat())
    try:
        await asyncio.sleep(0.005)  # heartbeat has entered its timed await
        with observed.phase_scope("send_1"):
            time.sleep(0.12)  # intentional test-only synchronous UI stall
            await asyncio.sleep(0.03)
        with pytest.raises(AssertionError, match="UI stall"):
            observed.assert_send_budgets()
    finally:
        observed.stop.set()
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("route", ["force", "cold"])
def test_cache_diagnostics_preserve_real_guarded_config_reads(monkeypatch, route):
    """Observe actual cold/forced reads without replacing enrolled callables."""
    from tldw_chatbook import config

    guarded = {
        name: getattr(config, name)
        for name in ("_load_settings_guarded", "_load_settings_uncached")
    }
    config._invalidate_config_caches()
    control = config.load_settings(force_reload=route == "force")
    observed = Observation()
    observed.install_config(monkeypatch, config)
    if route == "cold":
        config._invalidate_config_caches()
    actual = config.load_settings(force_reload=route == "force")
    assert actual == control
    assert all(getattr(config, name) is function for name, function in guarded.items())
