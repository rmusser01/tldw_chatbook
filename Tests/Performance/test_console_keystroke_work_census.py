"""Per-keystroke work census for the Console composer (TASK-24300, TASK-24301).

Wall clock is not usable as evidence on this surface, twice over. Textual's
``Pilot`` posts one callback per mounted widget per ``pause()`` -- 81,862
dispatch lookups for 40 keystrokes on a 488-widget screen -- so press latency
tracks widget count rather than app work. And the machine this repo is
developed on routinely carries a load average of 5-10 from concurrent agent
sessions; the same unchanged tree measured 3.80 and 6.75 ms/key for the same
input twenty minutes apart, which is a 78% swing with no code between the two
runs.

So these guards count CALLS, which are deterministic. Every number below
reproduced exactly on every run during the 2026-08-28 review, on a machine
whose wall-clock numbers moved by 3.5x.

The census that motivated the file, measured on dev ``3a3383123e`` with a
400-message conversation:

    messages_for_session   3.27 calls/key  ->  1,310 message snapshots per key

``messages_for_session`` materialises every stream buffer and deep-snapshots
every message. Four call sites used it as a predicate ("does this session have
any messages?") and one of them is on the composer keystroke path, so typing
degraded linearly with conversation length: 1.31 ms/key empty, 13.46 ms/key at
400 messages, and the whole difference was this call.
"""

from __future__ import annotations

from tldw_chatbook.UI.Console_Modules import context_spend as context_spend_module

import asyncio
import inspect
import json
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

import pytest

from Tests.private_profile import private_profile_test

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Printable keystrokes each census presses into the composer.
KEYSTROKES = 24

#: Messages seeded before the typing burst. Large enough that an O(N) term is
#: unmissable in a call count, small enough to keep the test quick.
SEEDED_MESSAGES = 200


#: The census profile's saved key (also the known-evidence connection's).
CENSUS_API_KEY = "sk-census-000000000000000000000000000000000000"


def _scratch_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, quiet_scheduler: bool = False
) -> None:
    """Seed the private-profile child's selected config with setup completed.

    A probe that skips this reads (and can write) the developer's real
    config; a probe that skips the API key lands Console in the SETUP state,
    where there is no composer to type into at all.

    Args:
        monkeypatch: pytest fixture used to set the environment.
        tmp_path: pytest fixture; the scratch tree's root.
        quiet_scheduler: Stretch the scheduler's 30 s poll to an hour. Its
            tick pays config and storage admissions (heartbeat and
            emergency-stop paths) and otherwise lands in whichever storage-
            unit phase is open 30 s after boot (TASK-33260).
    """
    del tmp_path  # The private-profile wrapper owns the selected scratch tree.
    config_file = Path(os.environ["TLDW_CONFIG_PATH"])
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        '[general]\nusers_name = "census"\n\n'
        "[first_run]\nsetup_completed = true\n\n"
        "[_first_run]\nsetup_completed = true\n\n"
        "[splash_screen]\nenabled = false\n\n"
        "[api_settings.openai]\n"
        f'api_key = "{CENSUS_API_KEY}"\n'
        + (
            "\n[scheduling]\nscheduler_poll_interval_seconds = 3600.0\n"
            if quiet_scheduler
            else ""
        )
    )
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "console_keystroke_census")
    from tldw_chatbook.config import load_settings

    load_settings(force_reload=True)


async def _settle(pilot: Any, passes: int = 30) -> None:
    """Let mount work finish so it is not billed to the typing burst."""
    for _ in range(passes):
        await asyncio.sleep(0.05)
        await pilot.pause()


#: The storage units TASK-33260 (PERF-01) added: what the two keystone
#: regressions of September 2026 actually cost, and what the derivation
#: counters above could not see (the census read 0 per key while every key
#: ran 27-69 guarded ``load_settings`` calls).
IO_UNITS = ("config_admissions", "storage_admissions", "helper_spawns", "os_opens")


def _census_log_path() -> Path | None:
    """Where to append this run's census, or None when nobody asked for it.

    ``TLDW_STORAGE_UNIT_CENSUS_LOG`` must name a file inside the runner's
    temporary directory (``RUNNER_TEMP`` on CI, else the system temporary
    directory); anything else, including a symlink that resolves outside it,
    is refused.

    Returns:
        The validated path, or None when the variable is unset.

    Raises:
        ValueError: The path lies outside the temporary directory.
    """
    requested = os.environ.get("TLDW_STORAGE_UNIT_CENSUS_LOG")
    if not requested:
        return None
    from tldw_chatbook.Utils.path_validation import validate_path

    root = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    return validate_path(requested, root)


def _report_census(case: str, census: dict[str, Any]) -> None:
    """Append one run's census to the requested log (see ``_census_log_path``).

    Args:
        case: The test case's node name.
        census: Phase name to its measured storage units.
    """
    census_log = _census_log_path()
    if census_log is None:
        return
    line = {"case": case, "platform": sys.platform, "census": census}
    if _STORAGE_UNIT_OBSERVER_RECEIPTS:
        receipt = _STORAGE_UNIT_OBSERVER_RECEIPTS[-1]
        diagnostic = receipt.get("credential_os_open_diagnostic")
        if diagnostic is not None:
            line["credential_os_open_diagnostic"] = {
                **diagnostic,
                "original_observer_complete": receipt["complete"],
                "original_observer_source_current": receipt["original_source_current"],
                "original_observer_hooks_retired_before_inactive": receipt[
                    "hooks_retired_before_inactive"
                ],
            }
    with open(census_log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


#: TASK-33802: who paid each storage admission and helper spawn billed to the
#: typing burst, so an over-ceiling burst names its caller in the failure.
_TYPING_BURST_CALLERS: list[str] = []
#: TASK-33644: why the GC census pass failed. The pass records rather than
#: asserts, so the census is still reported before the test fails on it.
_GC_PASS_FAILURES: list[str] = []
_STORAGE_UNIT_OBSERVER_RECEIPTS: list[dict[str, Any]] = []
#: The app package's own frames (not the venv's, whose path also names the
#: repository).
_APP_PACKAGE = str(Path(__file__).resolve().parents[2] / "tldw_chatbook") + os.sep


def _caller(unit: str) -> str:
    """One line per unit: the unit, the thread and the innermost app frames.

    Args:
        unit: The storage unit being billed.

    Returns:
        ``"<unit> on <thread>: file:line fn <- ..."`` for the last six
        ``tldw_chatbook`` frames, innermost first.
    """
    frames = [
        frame
        for frame in traceback.extract_stack()[:-3]
        if frame.filename.startswith(_APP_PACKAGE)
    ][-6:]
    where = " <- ".join(
        f"{frame.filename[len(_APP_PACKAGE) :]}:{frame.lineno} {frame.name}"
        for frame in reversed(frames)
    )
    return f"{unit} on {threading.current_thread().name}: {where or '(no app frame)'}"


_OS_OPEN_AUDIT: dict[str, Any] = {"installed": False, "bump": None}


def _count_os_open(event: str, args: tuple[Any, ...]) -> None:
    """Audit hook: ``os.open`` raises ``open`` with ``mode=None``."""
    if event == "open" and args[1] is None:
        bump = _OS_OPEN_AUDIT["bump"]
        if bump is not None:
            bump("os_opens")
        diagnostic = _OS_OPEN_AUDIT.get("credential_diagnostic")
        if (
            diagnostic is not None
            and diagnostic["counting"].get("on")
            and diagnostic["counting"].get("phase") == "idle"
        ):
            frame = code = namespace = None
            try:
                with diagnostic["lock"]:
                    if len(diagnostic["rows"]) >= 256:
                        diagnostic["overflow"] += 1
                        return
                chain = []
                frame = sys._getframe(1)
                for _ in range(32):
                    if frame is None or len(chain) >= 12:
                        break
                    code, namespace = frame.f_code, frame.f_globals
                    if code.co_filename.startswith(_APP_PACKAGE):
                        chain.append(
                            {
                                "file": code.co_filename[len(_APP_PACKAGE) :],
                                "qualname": code.co_qualname,
                                "line": frame.f_lineno,
                                "retained_original_code_globals_match": any(
                                    pin[1] is code and pin[2] is namespace
                                    for pin in diagnostic["observer"].pins
                                ),
                            }
                        )
                    frame = frame.f_back
                row = {
                    "phase": "idle",
                    "thread_id": threading.get_ident(),
                    "thread_name": threading.current_thread().name[:80],
                    "on_main_thread": threading.current_thread()
                    is threading.main_thread(),
                    "chain": chain,
                }
                with diagnostic["lock"]:
                    if len(diagnostic["rows"]) < 256:
                        diagnostic["rows"].append(row)
                    else:
                        diagnostic["overflow"] += 1
            except Exception as error:  # noqa: BLE001 -- diagnostic cannot replace original counting/error behavior
                with diagnostic["lock"]:
                    if len(diagnostic["invalid"]) < 8:
                        diagnostic["invalid"].append(type(error).__name__)
            finally:
                frame = code = namespace = None


def _count_storage_units(
    monkeypatch: pytest.MonkeyPatch, counts: dict[str, int], counting: dict[str, Any]
) -> Any:
    """Count original callback code locally, preserving stock source selection."""
    from tldw_chatbook.Backup_Recovery import config_participants, storage_admission
    from tldw_chatbook.DB.private_sqlite_process import HelperLease
    from Tests.Performance.console_storage_unit_observer import (
        OriginalStorageUnitObserver,
    )

    for key in IO_UNITS:
        counts[key] = 0
    lock = threading.Lock()

    def bump(key: str) -> None:
        if counting["on"]:
            with lock:
                counts[key] += 1
            if counting.get("burst") and key in ("storage_admissions", "helper_spawns"):
                _TYPING_BURST_CALLERS.append(_caller(key))

    observer = OriginalStorageUnitObserver(counts, counting, bump)
    try:
        observer.install(config_participants, storage_admission, HelperLease)
        if not _OS_OPEN_AUDIT["installed"]:
            sys.addaudithook(_count_os_open)
            _OS_OPEN_AUDIT["installed"] = True
        monkeypatch.setitem(_OS_OPEN_AUDIT, "bump", bump)
        diagnostic = {
            "counting": counting,
            "observer": observer,
            "lock": lock,
            "rows": [],
            "overflow": 0,
            "invalid": [],
        }
        counting["credential_diagnostic"] = diagnostic
        monkeypatch.setitem(_OS_OPEN_AUDIT, "credential_diagnostic", diagnostic)
    except BaseException:
        observer.close()
        raise
    return observer


def _capture_media_cleanup_timer(
    monkeypatch: pytest.MonkeyPatch, app_type: type[Any]
) -> list[Any]:
    """Capture the real startup cleanup callback for a separately billed phase.

    TASK-33802: its five-second timer can otherwise open a SQLite helper on
    an executor thread during typing. Keep the actual Textual timer and
    callback; only its automatic firing is held still, like trace maintenance.

    Args:
        monkeypatch: Fixture that restores the original timer factory.
        app_type: App class whose media-cleanup timer is being measured.

    Returns:
        Real startup callbacks to await in the cleanup phase.
    """
    callbacks: list[Any] = []
    real_set_timer = app_type.set_timer

    def set_timer(
        app: Any,
        delay: float,
        callback: Any = None,
        *,
        name: str | None = None,
        pause: bool = False,
    ) -> Any:
        if callback is not None and callback == app.perform_media_cleanup:
            callbacks.append(callback)
            pause = True
        return real_set_timer(app, delay, callback, name=name, pause=pause)

    monkeypatch.setattr(app_type, "set_timer", set_timer)
    return callbacks


@pytest.mark.asyncio
async def test_media_cleanup_capture_holds_only_its_timer(monkeypatch) -> None:
    """Other real timers fire; cleanup is awaited exactly once when driven."""
    from textual.app import App

    other_fired = asyncio.Event()
    cleanup_calls: list[None] = []

    class CleanupApp(App[None]):
        async def perform_media_cleanup(self) -> None:
            cleanup_calls.append(None)

        def on_mount(self) -> None:
            self.set_timer(0.001, self.perform_media_cleanup)
            self.set_timer(0.001, other_fired.set)

    callbacks = _capture_media_cleanup_timer(monkeypatch, CleanupApp)
    app = CleanupApp()
    async with app.run_test() as pilot:
        await asyncio.wait_for(other_fired.wait(), 5)
        await pilot.pause()
        assert len(callbacks) == 1
        assert not cleanup_calls
        await callbacks[0]()
        assert len(cleanup_calls) == 1


def _capture_native_pause_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, threading.Event]:
    """Hold wall-clock probes; retain the real callable and a drain barrier."""
    from tldw_chatbook.Backup_Recovery import storage_admission

    real_probe = storage_admission._local_pause_requested
    held = threading.Event()

    def held_probe() -> bool:
        held.set()
        return False

    monkeypatch.setattr(storage_admission, "_local_pause_requested", held_probe)
    return real_probe, held


async def _census_credential_ticks(
    console: Any, native_probe: Any, credential_interval: float
) -> None:
    """Drive the actual idle work at its configured cadence, without a clock."""
    from tldw_chatbook.Backup_Recovery.runtime_maintenance import (
        MAINTENANCE_PROBE_INTERVAL_SECONDS,
    )

    elapsed = 0.0
    for _ in range(IDLE_TICKS):
        console._poll_console_credential_readiness()
        elapsed += credential_interval
        while elapsed >= MAINTENANCE_PROBE_INTERVAL_SECONDS:
            assert not await asyncio.to_thread(
                native_probe
            ), "native maintenance requested"
            elapsed -= MAINTENANCE_PROBE_INTERVAL_SECONDS


@pytest.mark.asyncio
async def test_native_pause_capture_drains_and_bills_real_probes(monkeypatch, tmp_path):
    """Elapsed monitor time cannot leak opens; driven probes retain cost/result."""
    from tldw_chatbook.Backup_Recovery import runtime_maintenance, storage_admission

    entered, release = threading.Event(), threading.Event()
    probe_calls: list[None] = []
    requested = {"paused": False}
    path = tmp_path / "probe"
    path.write_bytes(b"probe")

    def probe() -> bool:
        if not probe_calls:
            entered.set()
            assert release.wait(5), "in-flight probe was not released"
        probe_calls.append(None)
        descriptor = os.open(path, os.O_RDONLY)
        os.close(descriptor)
        return requested["paused"]

    class Console:
        ticks = 0

        def _poll_console_credential_readiness(self) -> None:
            self.ticks += 1

    counts: dict[str, int] = {}
    counting = {"on": False}
    storage_observer = _count_storage_units(monkeypatch, counts, counting)
    monitor = None
    try:
        monkeypatch.setattr(storage_admission, "_local_pause_requested", probe)
        interval = runtime_maintenance.MAINTENANCE_PROBE_INTERVAL_SECONDS
        monkeypatch.setattr(
            runtime_maintenance, "MAINTENANCE_PROBE_INTERVAL_SECONDS", 0.01
        )
        monitor = asyncio.create_task(runtime_maintenance.monitor_app(object()))
        assert await asyncio.to_thread(entered.wait, 5)
        real_probe, held = _capture_native_pause_probe(monkeypatch)
        assert not held.is_set(), "hold completed before the in-flight probe"
        release.set()
        assert await asyncio.to_thread(held.wait, 5)
        assert len(probe_calls) == 1
        counting["on"] = True
        await asyncio.sleep(0.05)  # Several real monitor ticks remain held.
        assert counts["os_opens"] == 0
        monkeypatch.setattr(
            runtime_maintenance, "MAINTENANCE_PROBE_INTERVAL_SECONDS", interval
        )
        console = Console()
        await _census_credential_ticks(console, real_probe, interval / 4)
        assert console.ticks == IDLE_TICKS
        assert len(probe_calls) == 3
        assert counts["os_opens"] == 2
        requested["paused"] = True
        with pytest.raises(AssertionError, match="native maintenance requested"):
            await _census_credential_ticks(console, real_probe, interval / 4)
        assert counts["os_opens"] == 3
    finally:
        original_error = sys.exc_info()[1]
        try:
            counting["on"] = False
            release.set()
            if monitor is not None:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
        finally:
            receipt = storage_observer.close()
            _STORAGE_UNIT_OBSERVER_RECEIPTS.append(receipt)
            if not receipt["complete"]:
                if original_error is not None:
                    original_error.add_note(
                        "Original storage-unit observer evidence is incomplete."
                    )
                else:
                    raise AssertionError(
                        "Original storage-unit observer evidence is incomplete: "
                        + json.dumps(receipt)
                    )


async def _census(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    seeded_messages: int,
    *,
    storage_units: bool = False,
    known_evidence: bool = False,
) -> dict[str, int]:
    """Boot Console, seed a transcript, type, and return a call census.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
        seeded_messages: How many messages to append before typing.
        storage_units: Also count the burst in storage units (``IO_UNITS``
            keys) and then census the typing pause, the idle ticks and one
            warm Console visit (``<phase>:<unit>`` keys; see
            ``_census_idle_and_visit``). Holds the wall-clock loops still for
            the burst; off, the derivation census runs exactly as before.
        known_evidence: First settle a test result for the active connection
            into the app's shared evidence owner, so every readiness build
            looks it up and finds it (TASK-33005.2 AC#6).

    Returns:
        Mapping of counter name to calls observed during the typing burst
        (plus the storage-unit keys when requested).
    """
    storage_observer = None
    try:
        _scratch_env(monkeypatch, tmp_path, quiet_scheduler=storage_units)

        from textual.pilot import Pilot

        real_wait_for_screen = Pilot._wait_for_screen

        async def wait_for_screen(self: Any, timeout: float = 120.0) -> bool:
            # On the review Windows host the second full app's storage and
            # widget admission can exceed Textual's 30s default before typing
            # starts. This changes only the watchdog, never the work census.
            return await real_wait_for_screen(self, timeout=max(timeout, 120.0))

        monkeypatch.setattr(Pilot, "_wait_for_screen", wait_for_screen)

        from tldw_chatbook.app import TldwCli
        from tldw_chatbook.Chat.console_chat_models import (
            ConsoleChatMessage,
            ConsoleMessageRole,
        )
        from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
        from tldw_chatbook.UI.Console_Modules import session as session_module
        from tldw_chatbook.UI.Console_Modules import (
            console_spend_projection as spend_module,
        )
        from tldw_chatbook.UI.Screens import chat_screen as screen_module

        from tldw_chatbook.Chat import console_session_settings as settings_module

        counts: dict[str, int] = {
            "messages_for_session": 0,
            "snapshots": 0,
            "settings_readiness_builds": 0,
            "template_default_builds": 0,
            "snapshot_rows": 0,
            "spend_history_rows": 0,
            "cost_rows": 0,
            "cost_snapshot_rows": 0,
            "cost_projection_estimate_rows": 0,
            "context_rows": 0,
            "context_estimate_max_rows": 0,
            "cleanup_candidate_queries_completed": 0,
        }
        counting = {"on": False}

        real_messages_for_session = ConsoleChatStore.messages_for_session

        def counted_messages_for_session(
            self: ConsoleChatStore, session_id: str
        ) -> list[Any]:
            result = real_messages_for_session(self, session_id)
            if counting["on"]:
                counts["messages_for_session"] += 1
                counts["snapshots"] += len(result)
                caller = inspect.currentframe().f_back
                if caller is not None:
                    site = f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}"
                    key = f"snapshot_caller:{site}"
                    counts[key] = counts.get(key, 0) + 1
            return result

        monkeypatch.setattr(
            ConsoleChatStore, "messages_for_session", counted_messages_for_session
        )

        real_snapshot = ConsoleChatStore._snapshot

        def counted_snapshot(message: Any) -> Any:
            if counting["on"]:
                counts["snapshot_rows"] += 1
            return real_snapshot(message)

        monkeypatch.setattr(
            ConsoleChatStore, "_snapshot", staticmethod(counted_snapshot)
        )

        def _count_projected_rows(module: Any, name: str, key: str) -> None:
            real = getattr(module, name)

            def counted(messages: Any, *args: Any, **kwargs: Any) -> Any:
                if counting["on"]:
                    counts[key] += len(messages)
                return real(messages, *args, **kwargs)

            monkeypatch.setattr(module, name, counted)

        _count_projected_rows(
            spend_module, "build_console_spend_history_projection", "spend_history_rows"
        )
        _count_projected_rows(
            spend_module, "build_console_current_cost_messages", "cost_rows"
        )
        _count_projected_rows(
            spend_module, "build_console_context_messages", "context_rows"
        )
        _count_projected_rows(
            screen_module, "build_cost_snapshot", "cost_snapshot_rows"
        )
        _count_projected_rows(
            context_spend_module,
            "_estimate_tokens_locally",
            "cost_projection_estimate_rows",
        )
        real_context_estimate = context_spend_module.build_console_context_estimate

        def counted_context_estimate(messages: Any, *args: Any, **kwargs: Any) -> Any:
            if counting["on"]:
                # Textual may coalesce a different number of one-row draft
                # repaint calls in each mounted app. The largest input to any
                # call is the deterministic O(N) signal: 400 means the whole
                # transcript returned to the typing path.
                counts["context_estimate_max_rows"] = max(
                    counts["context_estimate_max_rows"], len(messages)
                )
            return real_context_estimate(messages, *args, **kwargs)

        monkeypatch.setattr(
            context_spend_module,
            "build_console_context_estimate",
            counted_context_estimate,
        )

        def _count_calls(module: Any, name: str, key: str) -> None:
            real = getattr(module, name)

            def counted(*args: Any, **kwargs: Any) -> Any:
                if counting["on"]:
                    counts[key] += 1
                return real(*args, **kwargs)

            monkeypatch.setattr(module, name, counted)

        # TASK-24301: the derivation legs. Patched on the modules the Console
        # session controller resolves them through, so a call that routes around
        # the memo is still seen.
        # TASK-33005 final review I-6: ChatScreen and the defaults module bind the
        # builder at import, so patching only its home module counted 0 forever.
        from tldw_chatbook.Chat import console_settings_defaults as defaults_module

        for module in (settings_module, screen_module, defaults_module):
            _count_calls(
                module, "build_console_settings_readiness", "settings_readiness_builds"
            )
        _count_calls(
            session_module,
            "default_console_session_settings",
            "template_default_builds",
        )

        trace_maintenance: list[tuple[Any, Any]] = []
        media_cleanup: list[Any] = []
        if storage_units:
            native_probe, probe_held = _capture_native_pause_probe(monkeypatch)
            storage_observer = _count_storage_units(monkeypatch, counts, counting)
            media_cleanup = _capture_media_cleanup_timer(monkeypatch, TldwCli)
            # The 1 Hz legacy trace-maintenance loop (armed 5 s after ready,
            # runs forever) is wall-clock driven: left running, a slow machine
            # bills more of its ticks to whatever is being measured. Captured
            # instead of scheduled; the ``trace`` phase bills it per tick.
            from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

            real_schedule = ConsoleRuntime._schedule_legacy_trace_maintenance
            monkeypatch.setattr(
                ConsoleRuntime,
                "_schedule_legacy_trace_maintenance",
                lambda runtime, database, normalizer_factory: trace_maintenance.append(
                    (database, normalizer_factory, runtime, real_schedule)
                ),
            )

        app = TldwCli()
        if storage_units:
            real_candidates = app.media_db.get_deletion_candidates

            def counted_candidates(*args: Any, **kwargs: Any) -> Any:
                result = real_candidates(*args, **kwargs)
                if counting["on"]:
                    counts["cleanup_candidate_queries_completed"] += 1
                return result

            monkeypatch.setattr(
                app.media_db, "get_deletion_candidates", counted_candidates
            )
        async with app.run_test(size=(170, 48)) as pilot:
            await _settle(pilot)
            if storage_units:
                # The serial monitor's held call drains an earlier native probe.
                assert await asyncio.to_thread(
                    probe_held.wait, 5
                ), "native probe never held"

            store = pilot.app.screen._ensure_console_chat_store()
            workspace_id = store.workspace_context.active_workspace_id
            session = store.ensure_session(title="census", workspace_id=workspace_id)
            # Restore the linear fixture in one pass. Appending 400 rows one at a
            # time repeatedly rebuilt the entire tree and made Windows setup
            # exceed the watchdog before the first measured keystroke.
            store._ingest_linear_messages(
                session.id,
                (
                    ConsoleChatMessage(
                        role=(
                            ConsoleMessageRole.USER
                            if index % 2 == 0
                            else ConsoleMessageRole.ASSISTANT
                        ),
                        content=f"census message {index} " + ("lorem ipsum " * 6),
                    )
                    for index in range(seeded_messages)
                ),
            )
            assert store.message_count(session.id) == seeded_messages
            await _settle(pilot, passes=10)

            # The fixture itself changed the transcript. Pay the legitimate cold
            # projection rebuild before the measured unchanged typing burst; a
            # real restored conversation also paints context and cost before input.
            screen = pilot.app.screen
            screen._context_spend._active_console_settings_context_estimate()
            screen._context_spend._build_console_cost_state()
            if known_evidence:
                _settle_known_connection_evidence(pilot.app, screen)
                await _settle(pilot, passes=10)  # Its one refresh is not typing.

            # The composer is the DEFAULT focus at rest; never call focus() here.
            # The first Input in walk order is a settings field, and a probe that
            # focuses it types into the wrong widget and measures nothing.
            assert type(pilot.app.focused).__name__ == "ConsoleComposerBar", (
                "census is only meaningful with the composer focused; got "
                f"{type(pilot.app.focused).__name__}"
            )

            # Hold the wall-clock timers still for the burst, as trace maintenance
            # is above. The 0.25 s credential poll builds readiness each tick
            # (billed per tick by the ``idle`` phase): left running, the slower
            # 400-message run billed more ticks to typing (34 vs 39 builds).
            if storage_units:
                credential_interval = screen._console_credential_poll_timer._interval
            screen._stop_console_credential_poll_timer()
            if storage_units:
                # The 0.2 s trailing draft-spend refresh, which a loaded machine
                # that leaves a >0.2 s gap between two presses fires mid-burst
                # (measured: 49 config admissions, not 27); the ``pause`` phase
                # fires it exactly once.
                screen._console_draft_spend_refresh.delay_seconds = 3600.0

            _TYPING_BURST_CALLERS.clear()
            counting["burst"] = True
            counting["on"] = True
            for _ in range(KEYSTROKES):
                await pilot.press("a")
            counting["on"] = False
            counting["burst"] = False

            if storage_units:
                await _census_idle_and_visit(
                    pilot,
                    counts,
                    counting,
                    trace_maintenance,
                    monkeypatch,
                    media_cleanup,
                    native_probe,
                    credential_interval,
                )

        return counts
    finally:
        if storage_observer is not None:
            original_error = sys.exc_info()[1]
            receipt = storage_observer.close()
            diagnostic = counting.get("credential_diagnostic")
            if diagnostic is not None:
                receipt["credential_os_open_diagnostic"] = {
                    "diagnostic_only": True,
                    "rows": diagnostic["rows"],
                    "overflow": diagnostic["overflow"],
                    "invalid": diagnostic["invalid"],
                    "max_rows": 256,
                    "max_chain_rows": 12,
                    "max_stack_walk": 32,
                    "frames_locals_arguments_results_retained": False,
                    "source_qualification_only_where_original_pin_matches": True,
                }
            _STORAGE_UNIT_OBSERVER_RECEIPTS.append(receipt)
            if not receipt["complete"]:
                if original_error is not None:
                    original_error.add_note(
                        "Original storage-unit observer evidence is incomplete."
                    )
                else:
                    raise AssertionError(
                        "Original storage-unit observer evidence is incomplete: "
                        + json.dumps(receipt)
                    )


def _settle_known_connection_evidence(app: Any, screen: Any) -> None:
    """Settle a reachable result for the census's openai connection.

    The poll absorbs the owner's new version once here, so the measured idle
    ticks bill only the steady-state lookup, not the one refresh it causes.
    """
    from tldw_chatbook.Chat.console_provider_endpoints import (
        effective_provider_endpoint,
    )
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        canonical_connection_identity,
    )
    from tldw_chatbook.Chat.provider_test_evidence import (
        ProviderDraftIdentity,
        ProviderProbeResult,
        ProviderTestEvidenceStore,
        connection_credential_revision,
    )

    store = ProviderTestEvidenceStore(lambda: app)
    identity = ProviderDraftIdentity(
        provider_key="openai",
        connection_identity=canonical_connection_identity(
            "openai", effective_provider_endpoint("openai", None, {})
        ),
        credential_source="stored",
        credential_revision=connection_credential_revision(CENSUS_API_KEY),
        draft_generation=0,
    )
    store.settle(store.begin(identity), ProviderProbeResult("reachable", ("gpt-4o",)))
    screen._poll_console_credential_readiness()
    assert screen._active_console_settings_readiness()[1].connection == identity, (
        "census evidence missed the active connection: every build would look "
        "it up and miss, so the evidence-hit cost would go unmeasured"
    )


#: Ticks each idle phase drives.
IDLE_TICKS = 8


async def _census_idle_and_visit(
    pilot: Any,
    counts: dict[str, int],
    counting: dict[str, Any],
    trace_maintenance: list[tuple[Any, Any, Any, Any]],
    monkeypatch: pytest.MonkeyPatch,
    media_cleanup: list[Any],
    native_probe: Any,
    credential_interval: float,
) -> None:
    """Census the typing pause, idle ticks and a warm visit in storage units.

    Every phase counts every thread from its start until its own timers
    settled and every worker it started finished, so nothing a phase begins
    lands in the next one. The wall-clock loops (credential poll, trace
    maintenance, native backup probe) are stopped/captured by ``_census``,
    so no phase depends on how many ticks a loaded machine fires.

    * ``pause:`` -- what the typing burst leaves behind once the user stops:
      the trailing draft-spend refresh (held off during the burst by
      ``_census``, fired once here) and every worker it and the burst's own
      trailing work started (the audit's 250-415 ms stall per typing pause).
    * ``idle:`` -- the 0.25 s credential poll, driven directly (per tick; the
      audit's 4 config admissions/s at rest), including real native backup
      probes at their production cadence (two per eight credential ticks).
    * ``trace:`` -- the 1 Hz legacy trace-maintenance batch, driven exactly
      as its loop does (per tick; a helper spawn/s, via
      ``run_owned_db_call``'s owned connection).
    * ``gc:`` -- the production maintenance loop's first GC pass
      (TASK-33644): its graph-epoch read, collection and compaction attempt.
    * ``cleanup:`` -- the captured startup media-cleanup callback, awaited
      once with its real SQLite candidate query and connection cleanup.
    * ``visit:`` -- Console -> Library (uncounted) -> Console. The route is
      reusable (TASK-31520), so the return is a warm resume: no mount, only
      ``on_screen_resume`` and what it schedules.
    * ``canary:`` -- one ``get_user_data_dir()`` call, which always admits:
      proof the wrapped seams still see the real config path.

    Blind spots: a NEW idle timer is not seen until it is driven here, and
    the 30 s scheduler tick is held still (``quiet_scheduler``), not billed.

    Args:
        pilot: the running app's pilot, on a settled Console after the burst.
        counts: census mapping; phase keys are added as ``<phase>:<unit>``.
        counting: shared ``{"on": bool}`` switch.
        trace_maintenance: one captured ``(database, normalizer_factory,
            runtime, real_schedule)`` per scheduling call: the arguments the
            runtime would have started its maintenance loop with, the runtime
            itself and the real ``_schedule_legacy_trace_maintenance`` that
            ``gc_pass`` uses to start that loop.
        monkeypatch: pytest fixture that owns the GC owned-call wrapper and
            ready delay; the pass restores the wrapper before it ends.
        media_cleanup: the captured real startup cleanup callback.
        native_probe: the real held native-pause callable, billed during idle.
        credential_interval: the captured real credential timer interval.
    """
    from textual import worker_manager

    from tldw_chatbook.Chat.console_trace_maintenance import LegacyTraceMaintenance
    from tldw_chatbook.DB.base_db import run_owned_db_call
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    app = pilot.app
    console = app.screen
    typing = dict(counts)
    phases: dict[str, int] = {}
    started: list[Any] = []
    real_new_worker = worker_manager.WorkerManager._new_worker

    def recording_new_worker(self: Any, *args: Any, **kwargs: Any) -> Any:
        worker = real_new_worker(self, *args, **kwargs)
        started.append(worker)
        return worker

    async def phase(name: str | None, body: Any) -> None:
        """Run ``body``, settle it, and bill it to ``name`` (None: uncounted)."""
        for unit in IO_UNITS:
            counts[unit] = 0
        started.clear()
        counting["on"] = name is not None
        counting["phase"] = name
        try:
            await body()
            await _settle(pilot, passes=20)
            # Drain every worker the phase started, including workers those
            # workers start while being awaited, so nothing leaks into (and is
            # billed to) the next phase. Not WorkerManager.wait_for_complete:
            # an empty list means "all". Bounded: a self-rearming worker ends
            # the drain instead of hanging it.
            drained = 0
            for _ in range(20):
                if drained == len(started):
                    break
                waits = [worker.wait() for worker in started[drained:]]
                drained = len(started)
                await asyncio.wait_for(
                    asyncio.gather(*waits, return_exceptions=True), 60
                )
                await _settle(pilot, passes=4)
        finally:
            counting["on"] = False
        if name is not None:
            phases.update({f"{name}:{unit}": counts[unit] for unit in IO_UNITS})
        if name == "cleanup":
            phases["cleanup:candidate_queries_completed"] = counts[
                "cleanup_candidate_queries_completed"
            ]

    async def typing_pause() -> None:
        spend_refresh = console._console_draft_spend_refresh
        spend_refresh.delay_seconds = 0.2
        spend_refresh.stop()
        spend_refresh.refresh()

    async def credential_ticks() -> None:
        await _census_credential_ticks(console, native_probe, credential_interval)

    assert trace_maintenance, "the Console never armed legacy trace maintenance"
    database, normalizer_factory, runtime, real_schedule = trace_maintenance[0]
    maintenance = LegacyTraceMaintenance(
        database, normalizer=normalizer_factory(), provider_active=lambda: False
    )
    # Capturing the scheduler leaves the fresh migration pending. Finish its
    # one-time write before measuring the settled idle completion checks.
    assert (await run_owned_db_call(database, maintenance.run_batch)).logical_complete

    async def trace_ticks() -> None:
        for _ in range(IDLE_TICKS):
            await run_owned_db_call(database, maintenance.run_batch)

    async def gc_pass() -> None:
        """Bill the maintenance loop's first GC pass, collection included.

        The census holds the loop back until here, so its first pass is the
        first GC of the profile: the graph-epoch read, the collection and the
        compaction attempt. On the census's small database compaction defers
        (``database_threshold``, a retryable reason) and the loop keeps that
        collection pending, so later passes are the epoch read plus a
        compaction retry with no collection -- the first pass is the superset,
        and the census asserts it ran exactly those three calls. It is billed
        from its epoch read through compaction (the batch normalization before
        it is the per-tick row above) and fails the census if a call raises or
        the collection is unusable. The 1 Hz backup-maintenance probe is held
        still for the whole phase: it walks every registered root (~37 opens)
        on its own thread, and landing inside the short billed window it
        doubled the pass's opens about one run in ten. The hold now covers
        every census phase; the idle phase pays the real native probe cost.
        """
        from tldw_chatbook.Chat import console_runtime as runtime_module

        real_owned = runtime_module.run_owned_db_call

        _GC_PASS_FAILURES.clear()
        window: dict[str, Any] = {"calls": [], "error": None}
        billed = asyncio.Event()
        # Owned calls run in worker threads that cancelling the loop does not
        # stop; cleanup waits for them before it restores anything.
        in_flight: set[asyncio.Future[Any]] = set()
        # Calls whose wrapper was cancelled; their outcome is read after
        # cleanup, since no wrapper is left to report it.
        orphaned: list[tuple[str, asyncio.Future[Any]]] = []

        async def owned(
            database_: Any, operation: Any, *args: Any, **kwargs: Any
        ) -> Any:
            name = getattr(operation, "__name__", "")
            if name == "current_graph_epoch" and not window["calls"]:
                for unit in IO_UNITS:
                    counts[unit] = 0
                counting["on"] = True
            if counting["on"] and not billed.is_set():
                window["calls"].append(name)
            call = asyncio.ensure_future(
                real_owned(database_, operation, *args, **kwargs)
            )
            in_flight.add(call)
            call.add_done_callback(in_flight.discard)
            try:
                result = await asyncio.shield(call)
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    orphaned.append((name, call))
                # Any call in the billed pass -- epoch read, collection or
                # compaction -- ends the census with its name, not a timeout.
                if window["calls"] and not billed.is_set():
                    counting["on"] = False
                    window["error"] = window["error"] or (
                        f"{name or 'an owned call'} raised {type(error).__name__}"
                    )
                    billed.set()
                raise
            if name == "collect" and window["calls"] and not billed.is_set():
                window["collected"] = getattr(result, "status", None)
            if name == "run_after_gc" and window["calls"] and not billed.is_set():
                counting["on"] = False
                # A deferral (e.g. database_threshold on the small census
                # database) is a whole pass: all three calls ran. A collection
                # that did not complete (e.g. a stale epoch) or was unusable
                # did partial work, so its counts are not the pass's.
                if window.get("collected") != "completed":
                    window["error"] = window["error"] or (
                        f"the billed collection ended {window.get('collected')!r}"
                    )
                elif getattr(result, "reason_code", "") == "logical_gc_unavailable":
                    window["error"] = "the billed pass found no usable collection"
                phases.update({f"gc:{unit}": counts[unit] for unit in IO_UNITS})
                billed.set()
            return result

        monkeypatch.setattr(runtime_module, "run_owned_db_call", owned)
        monkeypatch.setattr(
            runtime_module, "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS", 0.0
        )
        real_schedule(runtime, database, normalizer_factory)
        try:
            await asyncio.wait_for(billed.wait(), GC_PASS_WAIT_SECONDS)
        except TimeoutError:
            window["error"] = window["error"] or (
                f"the billed GC pass did not finish in {GC_PASS_WAIT_SECONDS} s"
            )
        finally:
            task = runtime._legacy_trace_maintenance_task
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                runtime._legacy_trace_maintenance_task = None
            if in_flight:
                await asyncio.wait(set(in_flight), timeout=GC_PASS_WAIT_SECONDS)
            monkeypatch.setattr(runtime_module, "run_owned_db_call", real_owned)
        late = [
            f"{name} then raised {type(call.exception()).__name__}"
            for name, call in orphaned
            if call.done() and not call.cancelled() and call.exception() is not None
        ]
        failure = "; ".join(filter(None, [window["error"], *late]))
        if not failure and window["calls"] != [
            "current_graph_epoch",
            "collect",
            "run_after_gc",
        ]:
            failure = f"the billed GC pass ran {window['calls']}"
        if failure:
            _GC_PASS_FAILURES.append(failure)
            # Unmeasured (never over a ceiling); the test fails on the
            # recorded failure once the census is reported.
            for unit in IO_UNITS:
                phases.setdefault(f"gc:{unit}", -1)

    async def navigate(target: str) -> None:
        await app.handle_screen_navigation(NavigateToScreen(target))

    async def canary() -> None:
        # Anti-vacuity: one guarded config read that always admits. A seam
        # that production stopped routing through reads 0 here, not a
        # green "paid down" everywhere else.
        from tldw_chatbook.config import get_user_data_dir
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        # A POSIX private-SQLite open starts a helper (ADR-125): prove its
        # start seam still counts. Windows uses native artifact handles and
        # still exercises the shared config/storage admission canary.
        connect_private_sqlite(
            "db.base", Path(get_user_data_dir()) / "census_canary.db"
        ).close()

    worker_manager.WorkerManager._new_worker = recording_new_worker
    try:
        assert len(media_cleanup) == 1, "startup media cleanup was not captured"
        await phase("cleanup", media_cleanup[0])
        await phase("pause", typing_pause)
        await phase("idle", credential_ticks)
        await phase("trace", trace_ticks)
        await gc_pass()
        await phase(None, lambda: navigate("library"))
        assert app.screen is not console, "census never left the Console"
        await phase("visit", lambda: navigate("chat"))
        await phase("canary", canary)
    finally:
        worker_manager.WorkerManager._new_worker = real_new_worker
    assert app.screen is console, "the reusable Console route built a new screen"
    # The phases also tick the derivation counters; hand back the burst's.
    counts.clear()
    counts.update(typing)
    counts.update(phases)


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_typing_never_snapshots_the_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """No keystroke materialises a transcript snapshot (TASK-24300).

    This is the guard the AC asks for: a predicate-shaped use of the
    snapshot API returning to the keystroke path fails here, and it fails on
    an EMPTY conversation too -- so the regression is caught before anyone
    has a long enough transcript to feel it.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
    """
    counts = await _census(monkeypatch, tmp_path, seeded_messages=SEEDED_MESSAGES)

    assert counts["messages_for_session"] == 0, (
        f"{counts['messages_for_session']} messages_for_session calls across "
        f"{KEYSTROKES} keystrokes ({counts['snapshots']} message snapshots; "
        f"callers={{{', '.join(f'{key}={value}' for key, value in counts.items() if key.startswith('snapshot_caller:'))}}} "
        "allocated). That call deep-copies the whole transcript, so any use "
        "of it on the keystroke path prices typing at O(conversation length) "
        "-- the TASK-24300 defect, which cost 12 ms per key at 400 messages. "
        "For an emptiness question use `has_messages`/`message_count`; to "
        "find the most recent match use `iter_messages_newest_first`."
    )


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_settled_400_message_typing_traverses_no_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Count actual mounted projection rows during unchanged long-chat typing."""
    counts = await _census(monkeypatch, tmp_path, seeded_messages=400)
    for key in (
        "messages_for_session",
        "snapshot_rows",
        "spend_history_rows",
        "cost_rows",
        "cost_snapshot_rows",
        "cost_projection_estimate_rows",
        "context_rows",
    ):
        assert counts[key] == 0, f"unchanged typing traversed {counts[key]} {key}"
    assert counts["context_estimate_max_rows"] <= 1


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_keystroke_work_does_not_scale_with_transcript_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Typing costs the same whether the conversation is empty or long.

    The absolute cost is deliberately not asserted -- see this module's
    docstring on why wall clock is not evidence here. What is asserted is
    that the census is IDENTICAL between the two transcript sizes, which is
    the property that actually broke: work that is O(N) in messages shows up
    as a difference between these two runs and nowhere else.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
    """
    empty = await _census(monkeypatch, tmp_path / "empty", seeded_messages=0)
    loaded = await _census(monkeypatch, tmp_path / "loaded", seeded_messages=400)
    request.node.user_properties.extend(
        [
            ("empty_census", json.dumps(empty, sort_keys=True)),
            ("400_message_census", json.dumps(loaded, sort_keys=True)),
        ]
    )

    # TASK-33374: ``context_estimate_max_rows`` is the largest input to any
    # context-estimate call, and whether Textual coalesces a one-row draft
    # repaint into the counting window is timing-dependent -- it read 1 for
    # the EMPTY transcript and 0 for 400 messages, the opposite of scaling.
    # The O(N) signal is that count reaching the transcript size, which the
    # <= 1 bounds below catch; every other key must match exactly.
    # TASK-33005 final review I-6: readiness builds (0 until every binding was
    # counted) ride the same trailing draft repaint: 25 and 25 on a quiet run,
    # 25 and 31-34 with six runs in parallel. Readiness never reads the
    # transcript, so they are bounded per key below instead.
    timing_bound = {"context_estimate_max_rows", "settings_readiness_builds"}
    exact_empty = {k: v for k, v in empty.items() if k not in timing_bound}
    exact_loaded = {k: v for k, v in loaded.items() if k not in timing_bound}
    assert exact_empty == exact_loaded, (
        f"per-keystroke work differs with transcript length: empty={empty}, "
        f"400 messages={loaded}. Something on the keystroke path is O(N) in "
        "the number of messages, which is what makes long conversations feel "
        "slower to type in than new ones."
    )
    assert empty["context_estimate_max_rows"] <= 1
    for census in (empty, loaded):
        builds = census["settings_readiness_builds"] / KEYSTROKES
        assert 0 < builds <= MAX_SETTINGS_READINESS_BUILDS_PER_KEY, census

    for key in (
        "snapshot_rows",
        "spend_history_rows",
        "cost_rows",
        "cost_snapshot_rows",
        "cost_projection_estimate_rows",
        "context_rows",
    ):
        assert (
            loaded[key] == 0
        ), f"typing traversed settled transcript in {key}: {loaded[key]} rows"
    assert loaded["context_estimate_max_rows"] <= 1


#: Per-keystroke ceilings for the Console state derivation (TASK-24301),
#: measured on dev `3a3383123e` before/after. Template defaults reach ZERO
#: because that derivation is memoised across passes; readiness stays live on
#: purpose (it reads `os.environ` for credentials, and caching it against a
#: stale snapshot is the task-177 regression).
MAX_SETTINGS_READINESS_BUILDS_PER_KEY = 3
MAX_TEMPLATE_DEFAULT_BUILDS_PER_KEY = 0


@pytest.mark.ui
@pytest.mark.asyncio
@private_profile_test
async def test_typing_does_not_rebuild_the_provider_derivation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A keystroke re-derives the provider graph at most a bounded number of times.

    Before TASK-24301 a single printable key ran the template-defaults builder
    3.25 times and the readiness builder 4.35 times, and threw every result
    away: the equality gate that follows skips the DOM write but not the
    compute, and nothing in the derivation moves between two characters of a
    word.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
    """
    counts = await _census(monkeypatch, tmp_path, seeded_messages=0)

    template_per_key = counts["template_default_builds"] / KEYSTROKES
    readiness_per_key = counts["settings_readiness_builds"] / KEYSTROKES

    assert template_per_key <= MAX_TEMPLATE_DEFAULT_BUILDS_PER_KEY, (
        f"{template_per_key:.2f} template-default builds per keystroke "
        f"(budget {MAX_TEMPLATE_DEFAULT_BUILDS_PER_KEY}). This derivation is "
        "a pure function of (app_config, provider, model) and is memoised "
        "across passes; a non-zero count means something bypassed the memo."
    )
    # Anti-vacuity (TASK-33005 final review I-6): measured 25 per 24 keys.
    assert readiness_per_key > 0, "no readiness build counted: an unpatched binding"
    assert readiness_per_key <= MAX_SETTINGS_READINESS_BUILDS_PER_KEY, (
        f"{readiness_per_key:.2f} readiness builds per keystroke (budget "
        f"{MAX_SETTINGS_READINESS_BUILDS_PER_KEY}). Readiness is deliberately "
        "NOT cached across passes -- it reads os.environ for credentials -- "
        "so the per-pass memo is the only thing keeping this bounded."
    )


#: TASK-33260 (PERF-01) storage-unit ceilings: RATCHETS, measured on
#: ``9cd9aad65f`` (see ``_census_idle_and_visit`` and ``_count_storage_units``
#: for what each phase and unit covers). Any increase fails; a PR that lowers
#: a count lowers its ceiling in the same commit. The TARGET is 0 per
#: keystroke, per typing pause and per idle tick. Paydown owners:
#:
#: * TASK-33265 (PERF-06) -- config warm-hit fast paths + Console derivation
#:   scopes: ``config_admissions`` on typing, the pause and the poll to 0.
#: * TASK-33267 (PERF-08) -- amortized storage admission (ADR-126
#:   amendment): ``storage_admissions`` and their ``os_opens`` directory walk.
#: * TASK-33268 (PERF-09) -- private-SQLite connection lifecycle:
#:   ``helper_spawns`` (TASK-33269, PERF-10, owns the trace-maintenance tick).
#:
#: Measured over 19 private-profile runs on macOS. ``config_admissions``
#: reproduce exactly. ``storage_admissions`` and ``helper_spawns`` jitter
#: DOWNWARD by 1-3 when a worker lands on an executor thread that still holds
#: a live connection (no connect admission, no helper), so these pins are the
#: observed maxima. ``os_opens`` also jitters UPWARD, in steps of ~37 opens
#: (one extra path walk) between runs with identical admission counts --
#: timing-dependent; TASK-33644 traced one such walk to the 1 Hz backup-
#: maintenance probe (``_local_pause_requested``) -- so its pins are the observed
#: maxima checked with ``OS_OPENS_JITTER_SLACK`` on top. TASK-33664.1 now
#: holds native wall-clock probes across all phases and bills their real cost
#: in idle at the production cadence, retaining these ceiling values; admission
#: counts are the exact signal. ``os_opens`` scales with
#: the depth of the profile path (one open per component): pinned on macOS's
#: ~11-component private-profile tmp path; Linux CI paths read lower.
MAX_TYPING_STORAGE_UNITS = {  # whole 24-key burst
    "config_admissions": 27,
    "storage_admissions": 54,
    "helper_spawns": 0,
    "os_opens": 19_224,
}
MAX_TYPING_PAUSE_STORAGE_UNITS = {
    "config_admissions": 22,
    "storage_admissions": 53,
    "helper_spawns": 3,
    "os_opens": 18_359,
}
MAX_CREDENTIAL_POLL_STORAGE_UNITS_PER_TICK = {
    "config_admissions": 1,
    "storage_admissions": 2,
    "helper_spawns": 0,
    "os_opens": 790.625,
}
MAX_TRACE_MAINTENANCE_STORAGE_UNITS_PER_TICK = {
    "config_admissions": 0,
    "storage_admissions": 2,
    "helper_spawns": 1,
    "os_opens": 715.375,
}
#: Visit admissions re-measured when #2888 was rebased onto dev 6423c4fbd1
#: (2026-09-29; pinned 37/107 at 9cd9aad65f). Config admissions are steady at
#: 39. Storage admissions read 110-115 across runs while workers started
#: during a phase could leak into the next; with every phase fully drained
#: they read 107-111, and the pin is that maximum + 1. The remaining jitter
#: is the 1 Hz legacy trace-maintenance tick and the credential poll landing
#: a varying number of ticks inside the visit window; PERF-10 (parks trace
#: maintenance) and PERF-06 (warm config reads) remove those sources and
#: tighten both numbers.
#: Re-pinned 39/112/35,351 -> 43/134/44,712 on 2026-09-30 (owner approved)
#: when #2888 was rebased onto dev 75c06af39a: #2922's Console hooks refresh
#: hook-permission state from disk on every visit (four runs: 43 config;
#: 127-133 storage, pinned max + 1; 42,830-44,712 opens). TASK-33642 took
#: that refresh off the visit path (a warm visit reuses the hook snapshot
#: while nothing it read changed) and restored 39/112/35,351 on 2026-10-03;
#: the same census then read 8 config, 56 storage, 9 helpers, ~2,160 opens
#: on macOS (dev at 2612fc56b2: 8, 59-60, 8-9, ~2,530).
MAX_VISIT_STORAGE_UNITS = {
    "config_admissions": 39,
    "storage_admissions": 112,
    "helper_spawns": 9,
    "os_opens": 35_351,
}
#: TASK-33644: the maintenance loop's first GC pass, billed on its own: the
#: graph-epoch read, the collection and the compaction attempt (deferred on
#: the census's small database). Pinned 2026-10-03 at dev 0409592a2d:
#: 0 / 9 / 3 / 54 in every run -- three owned database calls, each on a fresh
#: helper. ``os_opens`` is exact, not jitter: each helper start walks the
#: profile directory chain (one open per component, plus ``/`` and
#: ``/dev/null``), so it is ``3 x (components + 2)``: 54 under macOS's default
#: pytest temp dir (16 components), 39 on the Linux runner (11). A deeper
#: ``--basetemp`` adds 3 per component; a helper reused from a live
#: connection reads 18 fewer.
MAX_TRACE_GC_PASS_STORAGE_UNITS = {
    "config_admissions": 0,
    "storage_admissions": 9,
    "helper_spawns": 3,
    "os_opens": 54,
}
#: How long the GC census waits for its billed pass, and again for owned
#: calls still in their worker threads after cancelling the loop.
GC_PASS_WAIT_SECONDS = 60
#: Upward timing jitter allowance on ``os_opens`` only (see above).
OS_OPENS_JITTER_SLACK = 1.05
#: TASK-33643: os.open ceilings on the Linux perf-guard runner, where path
#: depth (and so opens per admission) differs from macOS -- the dicts above
#: hold macOS values. Pinned 2026-10-03 from three perf-guard runs (six
#: census lines, both evidence variants) at the highest value seen; the
#: shared jitter slack applies on top. Observed ranges: typing burst 54-81,
#: typing pause 153-206, credential poll 3.375-6.75 per tick, trace
#: maintenance 16.5-23.125 per tick, trace GC pass 26, visit 1,913-1,926.
#: The GC-pass row was re-pinned 26 -> 39 when the census started billing
#: the first pass (collection included): 3 helpers x (11 components + 2).
#: Every row assumes the runner's default pytest temp dir (11 path components
#: to the census profile): each directory walk opens one file per component,
#: so a run under a deeper ``--basetemp`` reads higher on every row by design.
#: The gate is for that CI layout, not for arbitrary local temp roots.
LINUX_OS_OPENS_CEILINGS = {
    "typing (whole burst)": 81,
    "typing pause": 206,
    "credential poll (per tick)": 6.75,
    "trace maintenance (per tick)": 23.125,
    "trace GC pass": 39,
    "visit": 1_926,
}


def _ceiling(phase: str, unit: str, ceilings: dict[str, float]) -> float | None:
    """The ceiling one census unit is gated on here, or None if ungated.

    Admissions and helper spawns count logic, so they gate everywhere.
    os.open counts depend on path depth, so each platform has its own:
    macOS uses the per-phase dicts, Linux ``LINUX_OS_OPENS_CEILINGS``, and
    any other platform is not gated on them.

    Args:
        phase: The census phase, a key of ``LINUX_OS_OPENS_CEILINGS``.
        unit: One of ``IO_UNITS``.
        ceilings: That phase's macOS ceilings.

    Returns:
        The ceiling, or None when this platform has no os.open pin.
    """
    if unit != "os_opens":
        return ceilings[unit]
    if sys.platform == "darwin":
        return ceilings[unit]
    if sys.platform.startswith("linux"):
        return LINUX_OS_OPENS_CEILINGS[phase]
    return None


def test_each_platform_gates_os_opens_on_its_own_ceilings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-33643: macOS and Linux each gate os.open on their own pins.

    Args:
        monkeypatch: Switches ``sys.platform``.
    """
    macos = {"config_admissions": 8, "os_opens": 2_000}
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _ceiling("visit", "os_opens", macos) == 2_000
    monkeypatch.setattr(sys, "platform", "linux")
    assert _ceiling("visit", "os_opens", macos) == LINUX_OS_OPENS_CEILINGS["visit"]
    assert _ceiling("visit", "config_admissions", macos) == 8
    monkeypatch.setattr(sys, "platform", "win32")
    assert _ceiling("visit", "os_opens", macos) is None
    assert _ceiling("visit", "config_admissions", macos) == 8


def test_the_census_log_stays_inside_the_runner_temp_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TASK-33643: the census writes only where the runner's temp dir allows.

    Args:
        monkeypatch: Sets the runner temp dir and the requested log path.
        tmp_path: The runner temp dir, and a sibling outside it.
    """
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.delenv("TLDW_STORAGE_UNIT_CENSUS_LOG", raising=False)
    assert _census_log_path() is None

    inside = runner_temp / "census.jsonl"
    monkeypatch.setenv("TLDW_STORAGE_UNIT_CENSUS_LOG", str(inside))
    assert _census_log_path() == inside.resolve()

    monkeypatch.setenv("TLDW_STORAGE_UNIT_CENSUS_LOG", str(tmp_path / "outside.jsonl"))
    with pytest.raises(ValueError):
        _census_log_path()

    escape = runner_temp / "escape.jsonl"
    escape.symlink_to(tmp_path / "outside.jsonl")
    monkeypatch.setenv("TLDW_STORAGE_UNIT_CENSUS_LOG", str(escape))
    with pytest.raises(ValueError):
        _census_log_path()


@pytest.mark.ui
@pytest.mark.asyncio
@pytest.mark.parametrize("known_evidence", [False, True], ids=["untested", "tested"])
@private_profile_test
async def test_console_storage_units_stay_within_their_ratchets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    known_evidence: bool,
) -> None:
    """Typing, an idle tick and a warm visit pay no more storage units than pinned.

    ``tested`` runs the same census with a settled test result for the active
    connection in the shared evidence owner (TASK-33005.2 AC#6): reading it
    must not raise a ceiling pinned without it, nor the readiness builds per
    keystroke. Each build's own provider-config reads with and without
    evidence are pinned equal by
    ``test_shared_evidence_adds_no_provider_config_reads``.

    The derivation counters above read 0 per key while each key ran 27-69
    guarded ``load_settings`` calls (the 2026-09-27 structural audit): this
    census counts the storage units themselves -- config admissions, storage
    admissions, private-SQLite helper spawns and ``os.open`` -- so the next
    per-call admission or per-connection helper cannot ship green.

    Args:
        monkeypatch: pytest fixture used for the scratch environment.
        tmp_path: pytest fixture; the scratch tree's root.
        request: pytest fixture; carries the census as user properties.
    """
    counts = await _census(
        monkeypatch,
        tmp_path,
        seeded_messages=0,
        storage_units=True,
        known_evidence=known_evidence,
    )
    measured = {
        "typing (whole burst)": (
            {unit: counts[unit] for unit in IO_UNITS},
            MAX_TYPING_STORAGE_UNITS,
        ),
        "typing pause": (
            {unit: counts[f"pause:{unit}"] for unit in IO_UNITS},
            MAX_TYPING_PAUSE_STORAGE_UNITS,
        ),
        "credential poll (per tick)": (
            {unit: counts[f"idle:{unit}"] / IDLE_TICKS for unit in IO_UNITS},
            MAX_CREDENTIAL_POLL_STORAGE_UNITS_PER_TICK,
        ),
        "trace maintenance (per tick)": (
            {unit: counts[f"trace:{unit}"] / IDLE_TICKS for unit in IO_UNITS},
            MAX_TRACE_MAINTENANCE_STORAGE_UNITS_PER_TICK,
        ),
        "trace GC pass": (
            {unit: counts[f"gc:{unit}"] for unit in IO_UNITS},
            MAX_TRACE_GC_PASS_STORAGE_UNITS,
        ),
        "visit": (
            {unit: counts[f"visit:{unit}"] for unit in IO_UNITS},
            MAX_VISIT_STORAGE_UNITS,
        ),
    }
    census: dict[str, Any] = {k: v[0] for k, v in measured.items()}
    cleanup_units = {unit: counts[f"cleanup:{unit}"] for unit in IO_UNITS}
    census["media cleanup"] = cleanup_units
    census["media cleanup candidate queries completed"] = counts[
        "cleanup:candidate_queries_completed"
    ]
    request.node.user_properties.append(("storage_units", json.dumps(census)))
    # TASK-33643: every run reports its census -- before any assertion, so a
    # failing run reports too -- and CI ceilings are pinned from those lines;
    # perf-guard.yml names the file and prints it.
    if _GC_PASS_FAILURES:
        census["trace GC pass failure"] = _GC_PASS_FAILURES[0]
    _report_census(request.node.name, census)
    assert not _GC_PASS_FAILURES, _GC_PASS_FAILURES[0]
    assert (
        counts["settings_readiness_builds"] / KEYSTROKES
        <= MAX_SETTINGS_READINESS_BUILDS_PER_KEY
    )
    request.node.user_properties.append(
        ("media_cleanup_units", json.dumps(cleanup_units))
    )
    assert (
        cleanup_units["storage_admissions"] >= 1
    ), "the startup cleanup phase did not reach its real database operation"
    assert (
        counts["cleanup:candidate_queries_completed"] == 1
    ), "startup cleanup did not complete exactly one real candidate query"
    # Windows validates SQLite artifacts with native handles in process;
    # POSIX uses HelperLease to preserve live advisory locks (ADR-125).
    if sys.platform != "win32":
        assert (
            cleanup_units["helper_spawns"] >= 1
        ), "the real startup cleanup query's cold SQLite helper was not counted"
    for unit in IO_UNITS:
        if sys.platform == "win32" and unit in {"helper_spawns", "os_opens"}:
            continue  # Native Windows handle opens have their own observer.
        assert counts[f"canary:{unit}"] >= 1, (
            f"census is blind: the canary (a guarded get_user_data_dir() plus "
            f"one private-SQLite open) counted 0 {unit}; the seam "
            "_count_storage_units wraps for it is no longer on the production "
            "path, so every ceiling below would pass vacuously."
        )
    slack = {"os_opens": OS_OPENS_JITTER_SLACK}
    over = [
        f"{phase} {unit}: {value} > ceiling {ceiling}"
        f"{' x ' + str(slack[unit]) if unit in slack else ''}"
        for phase, (values, ceilings) in measured.items()
        for unit, value in values.items()
        if (ceiling := _ceiling(phase, unit, ceilings)) is not None
        and value > ceiling * slack.get(unit, 1)
    ]
    # Only the units the burst went over: its ceilings allow dozens of
    # ordinary admissions, which would bury the one that broke it.
    burst_over = {
        unit
        for unit in ("storage_admissions", "helper_spawns")
        if f"typing (whole burst) {unit}:" in " ".join(over)
    }
    burst_callers = [
        entry
        for entry in _TYPING_BURST_CALLERS
        if entry.split(" on ", 1)[0] in burst_over
    ]
    callers = (
        " Typing-burst callers: " + " | ".join(burst_callers) + "."
        if burst_callers
        else ""
    )
    assert not over, (
        "Console storage units rose above their ratchet: "
        + "; ".join(over)
        + f". Full census: {json.dumps({k: v[0] for k, v in measured.items()})}."
        + callers
        + " "
        "Each unit is a real cost on a user path (an admission re-walks "
        "directory chains with one open() per component; a helper spawn is a "
        "python child, ~45-75 ms). Take the new cost off the path; never "
        "raise a ceiling (see MAX_TYPING_STORAGE_UNITS for the owners)."
    )
