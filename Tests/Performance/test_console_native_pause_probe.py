"""Native captured-send regression budgets with passive original-code diagnostics.

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
import hashlib
import inspect
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


# Recorded before the final integration measurement (TASK-34561). Earlier native
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
        self.phase_windows = {}
        self.delay_windows = []
        self.stop = threading.Event()
        self.passive_seams = None
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

    def _start_seams(self, *, config_only=False):
        from Tests.Performance.console_native_pause_seams import PassivePauseProbeSeams

        assert self.passive_seams is None, "one observer owns one monitoring tool"
        self.passive_seams = PassivePauseProbeSeams(self, config_only=config_only)
        self.passive_seams.start()

    def stop_seams(self):
        """Remove owned local hooks once, including standalone observer tests."""
        if self.passive_seams is not None and self.passive_seams.active:
            self.passive_seams.stop()

    def install_config(self, monkeypatch, config):
        """Observe original config bodies without replacing any public alias."""
        assert sys.modules.get("tldw_chatbook.config") is config
        self._start_seams(config_only=True)

    def install(self, monkeypatch):
        # Preload the same original defining modules before selecting local codes.
        from tldw_chatbook import config  # noqa: F401
        from tldw_chatbook.Backup_Recovery import config_participants, storage_admission  # noqa: F401
        from tldw_chatbook.DB.private_sqlite_process import HelperLease  # noqa: F401

        if os.name == "nt":
            from tldw_chatbook.Utils.windows_files import _Native  # noqa: F401

        self._start_seams()

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
                passive_seam_observer=(
                    self.passive_seams.receipt()
                    if self.passive_seams is not None
                    else None
                ),
                heartbeat={
                    p: dict(
                        count=len(v),
                        max_seconds=max(v, default=0),
                        over_100ms=sum(x > 0.1 for x in v),
                    )
                    for p in self.phases.keys() | self.lags.keys()
                    for v in (self.phase_lags(p),)
                },
                limits="One native run per host. Passive original-code observation overhead. Pilot/headless dispatch is included in wall time. Nested seam times overlap. Normal app timers and startup trace maintenance remain enabled. Heartbeat delay intervals are intersected with phase windows, including synchronous entry stalls. Config metrics observe original code through all imported aliases. Attempt counts use original START; elapsed timing gaps remain explicit. All production callable identities remain installed. No real provider, OS keyring or model download.",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")


async def _await_probe_trace_settlement(store, *, deadline):
    """Await original scheduler retirement within the captured-send allowance."""
    while store.pending_provider_trace_settlement_work_count():
        remaining = deadline - time.perf_counter()
        assert (
            remaining > 0
        ), "Original provider trace settlement did not retire within send budget"
        await asyncio.sleep(min(0.01, remaining))


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
    result = {"complete": False, "provider_calls": 0}
    heartbeat = None
    try:
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
        heartbeat = asyncio.create_task(observed.heartbeat())
        app = TldwCli()
        async with app.run_test(size=(140, 42)) as pilot:
            # The existing observer and heartbeat include the original initial
            # task's receipt preparation. Keep this prerequisite inside the
            # original 900s deadline before asserting the completed screen.
            while not getattr(app, "_initial_screen_pushed", False):
                await asyncio.sleep(0.01)
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
            custody = []

            def observe_custody(submission, turn_id):
                record = runtime._turn_custody[turn_id]
                assert record.session_id == submission.session_id
                assert record.turn_id == turn_id == submission.turn_id
                assert isinstance(record.task, asyncio.Task)
                hooks = screen._hooks
                custody.append(
                    (
                        submission.session_id,
                        turn_id,
                        record.task,
                        hooks,
                        hooks.pending_send_identity,
                        time.perf_counter(),
                    )
                )
                tasks.append(record.task)

            original_accept = runtime.accept_turn

            def accept(request, **kwargs):
                turn_id = original_accept(request, **kwargs)
                observe_custody(request, turn_id)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", accept)
            original_receive = runtime.accept_received_intent

            def receive(intent, **kwargs):
                turn_id = original_receive(intent, **kwargs)
                observe_custody(intent, turn_id)
                return turn_id

            monkeypatch.setattr(runtime, "accept_received_intent", receive)
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
            trace_store = controller.store
            for index in range(1, 4):
                send_deadline = time.perf_counter() + MAX_CAPTURED_SEND_SECONDS
                with observed.phase_scope(f"send_{index}"):
                    screen._session._sync_console_session_draft()
                    composer.load_draft(f"Native pause probe message {index}")
                    before = len(tasks)
                    session_id = controller.store.active_session_id
                    hooks = screen._hooks
                    await screen._send_console_message_from_visible_action(
                        session_id=session_id
                    )
                    pending = None
                    if len(tasks) == before:
                        pending = hooks.pending_send_identity
                        assert (
                            pending is not None and pending[0] == session_id
                        ), "send never reached turn custody or a pending hook worker"
                        # Hook dispatch may return before its worker admits the
                        # turn. Observe only that exact Send within its old budget.
                        while len(tasks) == before:
                            assert screen.is_mounted and runtime.view is screen
                            assert screen._hooks is hooks
                            assert controller.store.active_session_id == session_id
                            assert (
                                hooks.pending_send_identity == pending
                            ), "pending Send cleared or changed without turn custody"
                            assert (
                                time.perf_counter() < send_deadline
                            ), "pending Send never reached turn custody within budget"
                            await asyncio.sleep(0.01)
                    assert len(tasks) == before + 1, "send never reached turn custody"
                    (
                        accepted_session,
                        turn_id,
                        task,
                        owner,
                        identity,
                        accepted_at,
                    ) = custody[-1]
                    assert accepted_session == session_id and task is tasks[-1]
                    assert all(prior[1] != turn_id for prior in custody[:before])
                    if pending is not None:
                        assert owner is hooks and identity == pending
                        assert (
                            accepted_at <= send_deadline
                        ), "pending Send reached turn custody after its budget"
                    await asyncio.wait_for(asyncio.shield(task), 180)
                    assert controller.run_state.status is ConsoleRunStatus.COMPLETED
                    await screen._sync_native_console_chat_ui()
                    await asyncio.sleep(0.25)
                    assert controller.store is trace_store
                    await _await_probe_trace_settlement(
                        trace_store, deadline=send_deadline
                    )
                    assert controller.store is trace_store
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
        try:
            observed.stop.set()
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            sampler = getattr(observed, "sampler", None)
            if sampler is not None:
                sampler.join(timeout=2)
        finally:
            try:
                observed.stop_seams()
            finally:
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
    try:
        if route == "cold":
            config._invalidate_config_caches()
        actual = config.load_settings(force_reload=route == "force")
        assert actual == control
        assert all(
            getattr(config, name) is function for name, function in guarded.items()
        )
    finally:
        observed.stop_seams()


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["early", "late"])
@private_profile_test
async def test_cache_observer_preserves_checked_display_for_real_import_orders(
    monkeypatch, tmp_path, request, record_property, order
):
    """Passive observation keeps the real issued display contract in either order."""
    from tldw_chatbook import config

    module_name = "tldw_chatbook.UI.Screens.chat_screen"
    assert module_name not in sys.modules, "control must start before screen import"
    original_loader = config.load_settings
    guarded = {
        name: getattr(config, name)
        for name in ("_load_settings_guarded", "_load_settings_uncached")
    }
    if order == "early":
        from Tests.UI import test_console_checked_display_scope as actual

        assert sys.modules[module_name].load_settings is original_loader
    observed = Observation()
    observed.install_config(monkeypatch, config)
    try:
        if order == "late":
            from Tests.UI import test_console_checked_display_scope as actual

        screen_module = sys.modules[module_name]
        database, _store, _controller, screen, tasks = actual._screen(tmp_path)
        rendered, enclosing, before_nested, serialized = [], [], [], []
        try:
            projection = await actual._warm(screen, tasks)

            def render():
                rendered.append(screen._provider_readiness_app_config())
                enclosing.append(getattr(actual.raw._local, "operation", None))

            with actual._actual_calls() as warm_calls:
                for _ in range(6):
                    assert actual.ChatScreen._run_console_config_sync(screen, render)

            def nested_read():
                before_nested.append(getattr(actual.raw._local, "operation", None))
                serialized.append(config.read_cli_config_serialized())

            with actual._actual_calls() as nested_calls:
                assert actual.ChatScreen._run_console_config_sync(screen, nested_read)
            record_property("import_order", order)
            record_property("main_config_entries", warm_calls["main_scopes"])
            record_property("main_native_opens", warm_calls["main_opens"])
            record_property("nested_config_entries", nested_calls["main_scopes"])
            assert all(
                getattr(config, name) is value for name, value in guarded.items()
            )
            assert screen_module.load_settings is config.load_settings
            assert projection._display_proof is not None
            assert len(rendered) == 6 and all(
                value is projection.value for value in rendered
            )
            assert warm_calls["main_scopes"] == 0, warm_calls
            assert warm_calls["main_opens"] == 0, warm_calls
            assert enclosing == [None] * 6
            assert before_nested == [None], "nested reader borrowed display authority"
            assert len(serialized) == 1 and isinstance(serialized[0], str)
            assert nested_calls["main_scopes"] == 3, nested_calls
            assert len(set(nested_calls["owners"])) == 1, nested_calls
            assert all(
                owner not in actual.raw._states for owner in nested_calls["owners"]
            )
            assert nested_calls["main_opens"] > 0, nested_calls
            assert len(nested_calls["disk_reads"]) == 1
            assert nested_calls["disk_reads"][0] is not None
            assert nested_calls["disk_reads"][0] not in actual.raw._states
            assert not any(
                state.source is config for state in actual.raw._states.values()
            )
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)
            database.close()
    finally:
        observed.stop_seams()


@pytest.mark.asyncio
@private_profile_test
async def test_passive_probe_preserves_actual_stock_sensitive_bundle(
    monkeypatch, tmp_path, request, record_property
):
    """The full observer must not select the preceding custom sensitive route."""
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import config_participants as life
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils import sensitive_paths as sensitive

    assert "tldw_chatbook.app" not in sys.modules
    originals = life.operation, storage._acquire_storage, config.load_settings
    assert (
        sensitive._stock_sensitive_config_bundle(sensitive._raw_inputs_key()[2])
        is not None
    )
    observed = Observation()
    try:
        observed.install(monkeypatch)
        assert (
            sensitive._stock_sensitive_config_bundle(sensitive._raw_inputs_key()[2])
            is not None
        )
        assert life.operation is originals[0]
        assert storage._acquire_storage is originals[1]
        assert config.load_settings is originals[2]
        assert observed.passive_seams.bindings_current()
    finally:
        observed.stop.set()
        sampler = getattr(observed, "sampler", None)
        try:
            if sampler is not None:
                sampler.join(timeout=2)
                assert not sampler.is_alive()
        finally:
            observed.stop_seams()
    receipt = observed.passive_seams.receipt()
    assert (
        receipt["restoration"] == "selected_local_callbacks_removed_tool_freed_global0"
    )
    assert receipt["original_bindings_and_bodies_unchanged"]
    assert receipt["overflow"] == 0
    assert (
        sensitive._stock_sensitive_config_bundle(sensitive._raw_inputs_key()[2])
        is not None
    )
    assert "tldw_chatbook.app" not in sys.modules
    record_property("passive_observer", json.dumps(receipt, sort_keys=True))


@pytest.mark.asyncio
@private_profile_test
async def test_passive_probe_counts_original_finite_storage_acquisition(
    monkeypatch, tmp_path, request, record_property
):
    """Independent original-code call events agree on real admission attempts."""
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    assert "tldw_chatbook.app" not in sys.modules
    acquisition = storage._acquire_storage
    acquisition_code = acquisition.__code__
    native_code = None
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import _Native

        native_code = inspect.getattr_static(_Native, "open_handle").__code__
    independent = collections.Counter()
    previous_profile = sys.getprofile()
    assert previous_profile is None, "this finite control owns its passive counter"

    def calls(frame, event, value):
        if event == "call":
            if frame.f_code is acquisition_code:
                independent["acquisition"] += 1
            elif native_code is not None and frame.f_code is native_code:
                independent["native"] += 1

    def census():
        with storage._lock:
            return tuple(
                frozenset(values)
                for values in (
                    storage._live_leases,
                    storage._pending_acquisitions,
                    storage._operations,
                    storage._retiring_holds,
                )
            )

    def count(label):
        return sum(
            value[0]
            for (phase, thread, seam, _), value in observed.rows.items()
            if phase == "finite_control" and thread == "main" and seam == label
        )

    observed, lease = Observation(), None
    before = census()
    try:
        observed.install(monkeypatch)
        observed.phase = "finite_control"
        sys.setprofile(calls)
        try:
            lease = storage.acquire_storage()
            assert lease in storage._live_leases
        finally:
            try:
                if lease is not None:
                    lease.close()
            finally:
                sys.setprofile(previous_profile)
        assert independent["acquisition"] == 1
        assert (
            count(storage.__name__ + "._acquire_storage") == independent["acquisition"]
        )
        if os.name == "nt":
            assert independent["native"] > 0
            assert count("_Native.open_handle") == independent["native"]
        assert census() == before
        assert storage._acquire_storage is acquisition
        assert observed.passive_seams.bindings_current()
    finally:
        try:
            sys.setprofile(previous_profile)
            if lease is not None:
                lease.close()
        finally:
            observed.stop.set()
            sampler = getattr(observed, "sampler", None)
            try:
                if sampler is not None:
                    sampler.join(timeout=2)
                    assert not sampler.is_alive()
            finally:
                observed.stop_seams()
    assert sys.getprofile() is previous_profile
    assert census() == before
    assert observed.passive_seams.receipt()["restoration"] is not None
    assert "tldw_chatbook.app" not in sys.modules
    record_property(
        "independent_original_call_counts", json.dumps(independent, sort_keys=True)
    )
