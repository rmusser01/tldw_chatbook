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


#: TASK-33802: who paid each storage admission and helper spawn billed to the
#: typing burst, so an over-ceiling burst names its caller in the failure.
_TYPING_BURST_CALLERS: list[str] = []
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
        f"{frame.filename[len(_APP_PACKAGE):]}:{frame.lineno} {frame.name}"
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


def _count_storage_units(
    monkeypatch: pytest.MonkeyPatch, counts: dict[str, int], counting: dict[str, Any]
) -> None:
    """Count the ADR-125/ADR-126 storage units by wrapping their real seams.

    Harness-only: every wrapper calls straight through, so production
    behaviour is unchanged. Counts every thread, because a unit paid on a
    worker still costs the user a core and the GIL; the phases that use
    these counters stop the wall-clock-driven credential poll so a slow
    machine cannot bill extra ticks to the unit being measured.

    * ``config_admissions`` -- OUTERMOST ``config_participants.operation``
      entries per thread (nested ones reuse the open scope and pay nothing).
      Every guarded config reader routes through the module attribute.
    * ``storage_admissions`` -- ``storage_admission._acquire_storage``, the
      one module-global every ``acquire_storage`` call reaches however the
      caller imported the public name.
    * ``helper_spawns`` -- ``HelperLease.start``: one ``python -I -S``
      private-SQLite helper child each.
    * ``os_opens`` -- ``os.open``, the admission directory walk's unit (one
      per path component), counted from the ``open`` audit event (``mode``
      is ``None`` only for ``os.open``; ``builtins.open`` is not counted).
      Never by replacing ``os.open``: the raw participants require
      ``os.open in os.supports_dir_fd``, so a wrapper turns every config
      read into ``RecoveryRequired('raw_source_selection_changed')``.

    Args:
        monkeypatch: pytest fixture that owns (and undoes) the wrappers.
        counts: census mapping; the ``IO_UNITS`` keys are added here.
        counting: shared ``{"on": bool}`` switch.
    """
    import contextlib
    import threading

    from tldw_chatbook.Backup_Recovery import config_participants, storage_admission
    from tldw_chatbook.DB.private_sqlite_process import HelperLease

    for key in IO_UNITS:
        counts[key] = 0
    lock = threading.Lock()
    depth = threading.local()

    def bump(key: str) -> None:
        if counting["on"]:
            with lock:
                counts[key] += 1
            if counting.get("burst") and key in ("storage_admissions", "helper_spawns"):
                _TYPING_BURST_CALLERS.append(_caller(key))

    real_operation = config_participants.operation

    @contextlib.contextmanager
    def counted_operation(*args: Any, **kwargs: Any) -> Any:
        level = getattr(depth, "level", 0)
        if level == 0:
            bump("config_admissions")
        depth.level = level + 1
        try:
            with real_operation(*args, **kwargs) as active:
                yield active
        finally:
            depth.level = level

    monkeypatch.setattr(config_participants, "operation", counted_operation)

    real_acquire = storage_admission._acquire_storage

    def counted_acquire(*args: Any, **kwargs: Any) -> Any:
        bump("storage_admissions")
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(storage_admission, "_acquire_storage", counted_acquire)

    real_start = HelperLease.start

    def counted_start(cls: type, *args: Any, **kwargs: Any) -> Any:
        bump("helper_spawns")
        return real_start(*args, **kwargs)

    monkeypatch.setattr(HelperLease, "start", classmethod(counted_start))

    if not _OS_OPEN_AUDIT["installed"]:
        # Audit hooks cannot be removed; one per process, re-aimed per census.
        sys.addaudithook(_count_os_open)
        _OS_OPEN_AUDIT["installed"] = True
    monkeypatch.setitem(_OS_OPEN_AUDIT, "bump", bump)


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

    monkeypatch.setattr(ConsoleChatStore, "_snapshot", staticmethod(counted_snapshot))

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
    _count_projected_rows(screen_module, "build_cost_snapshot", "cost_snapshot_rows")
    _count_projected_rows(
        screen_module, "_estimate_tokens_locally", "cost_projection_estimate_rows"
    )
    real_context_estimate = screen_module.build_console_context_estimate

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
        screen_module, "build_console_context_estimate", counted_context_estimate
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
    if storage_units:
        _count_storage_units(monkeypatch, counts, counting)
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
    async with app.run_test(size=(170, 48)) as pilot:
        await _settle(pilot)

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
        screen._active_console_settings_context_estimate()
        screen._build_console_cost_state()
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
                pilot, counts, counting, trace_maintenance, monkeypatch
            )

    return counts


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
) -> None:
    """Census the typing pause, idle ticks and a warm visit in storage units.

    Every phase counts every thread from its start until its own timers
    settled and every worker it started finished, so nothing a phase begins
    lands in the next one. The wall-clock loops (credential poll, trace
    maintenance) are stopped/captured by ``_census``, so no phase depends on
    how many ticks a loaded machine fires.

    * ``pause:`` -- what the typing burst leaves behind once the user stops:
      the trailing draft-spend refresh (held off during the burst by
      ``_census``, fired once here) and every worker it and the burst's own
      trailing work started (the audit's 250-415 ms stall per typing pause).
    * ``idle:`` -- the 0.25 s credential poll, driven directly (per tick; the
      audit's 4 config admissions/s at rest).
    * ``trace:`` -- the 1 Hz legacy trace-maintenance batch, driven exactly
      as its loop does (per tick; a helper spawn/s, via
      ``run_owned_db_call``'s owned connection).
    * ``gc:`` -- one eligible GC interval of the production maintenance loop
      after it parked (TASK-33644): from the pass's graph-epoch read through
      its collection and compaction.
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
        trace_maintenance: the captured ``(database, normalizer_factory)``
            the runtime would have started its maintenance loop with.
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

    async def typing_pause() -> None:
        spend_refresh = console._console_draft_spend_refresh
        spend_refresh.delay_seconds = 0.2
        spend_refresh.stop()
        spend_refresh.refresh()

    async def credential_ticks() -> None:
        for _ in range(IDLE_TICKS):
            console._poll_console_credential_readiness()

    assert trace_maintenance, "the Console never armed legacy trace maintenance"
    database, normalizer_factory, runtime, real_schedule = trace_maintenance[0]
    maintenance = LegacyTraceMaintenance(
        database, normalizer=normalizer_factory(), provider_active=lambda: False
    )

    async def trace_ticks() -> None:
        for _ in range(IDLE_TICKS):
            await run_owned_db_call(database, maintenance.run_batch)

    async def gc_pass() -> None:
        """Bill one eligible GC interval of the real maintenance loop.

        The loop's first pass collects at once (uncounted); the moment it
        returns, the GC interval becomes too long to reach, so the loop parks
        and stays parked. The graph epoch is then advanced with no exchange
        signal -- so the next pass is eligible whether or not a collection is
        still pending -- and billing is armed; only then does the interval
        drop to zero, so the next park poll wakes exactly one GC pass. On the
        census's small database the first compaction defers
        (``database_threshold``, a retryable reason), so the loop keeps that
        collection pending and the billed pass is the epoch read plus the
        compaction retry -- the steady state of an idle small profile. That pass
        is billed from its epoch read through compaction; it fails the census
        if compaction raises or the collection is unusable.
        """
        from tldw_chatbook.Chat import console_runtime as runtime_module

        real_owned = runtime_module.run_owned_db_call
        boot_passes: list[str] = []
        window: dict[str, Any] = {"armed": False, "started": False, "error": None}
        billed = asyncio.Event()

        async def owned(database_: Any, operation: Any, *args: Any, **kwargs: Any) -> Any:
            name = getattr(operation, "__name__", "")
            if name == "current_graph_epoch" and window["armed"] and not window["started"]:
                window["started"] = True
                for unit in IO_UNITS:
                    counts[unit] = 0
                counting["on"] = True
            try:
                result = await real_owned(database_, operation, *args, **kwargs)
            except BaseException as error:
                # Any call in the billed pass -- epoch read, collection or
                # compaction -- ends the census with its name, not a timeout.
                if window["started"] and not billed.is_set():
                    counting["on"] = False
                    window["error"] = f"{name or 'an owned call'} raised {type(error).__name__}"
                    billed.set()
                raise
            if name == "run_after_gc":
                if not window["started"]:
                    boot_passes.append(name)
                    # Hold the loop parked until the billed pass is armed.
                    monkeypatch.setattr(
                        runtime_module, "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS", 1e9
                    )
                elif not billed.is_set():
                    counting["on"] = False
                    # A deferral (e.g. database_threshold on the small census
                    # database) is a whole pass: all three calls ran. Only a
                    # pass whose collection was unusable is not.
                    if getattr(result, "reason_code", "") == "logical_gc_unavailable":
                        window["error"] = "the billed pass found no usable collection"
                    phases.update({f"gc:{unit}": counts[unit] for unit in IO_UNITS})
                    billed.set()
            return result

        def advance_graph_epoch() -> None:
            with database.transaction() as cursor:
                cursor.execute(
                    "UPDATE console_trace_graph_epoch SET epoch = epoch + 1"
                    " WHERE singleton_id = 1"
                )

        monkeypatch.setattr(runtime_module, "run_owned_db_call", owned)
        monkeypatch.setattr(
            runtime_module, "LEGACY_TRACE_MAINTENANCE_READY_DELAY_SECONDS", 0.0
        )
        monkeypatch.setattr(
            runtime_module, "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS", 0.0
        )
        real_schedule(runtime, database, normalizer_factory)
        try:
            for _ in range(600):
                if boot_passes:
                    break
                await asyncio.sleep(0.05)
            assert boot_passes, "the maintenance loop never ran its first GC pass"
            await real_owned(database, advance_graph_epoch)
            window["armed"] = True
            monkeypatch.setattr(
                runtime_module, "TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS", 0.0
            )
            await asyncio.wait_for(billed.wait(), 60)
            assert window["error"] is None, window["error"]
        finally:
            task = runtime._legacy_trace_maintenance_task
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                runtime._legacy_trace_maintenance_task = None
            monkeypatch.setattr(runtime_module, "run_owned_db_call", real_owned)

    async def navigate(target: str) -> None:
        await app.handle_screen_navigation(NavigateToScreen(target))

    async def canary() -> None:
        # Anti-vacuity: one guarded config read that always admits. A seam
        # that production stopped routing through reads 0 here, not a
        # green "paid down" everywhere else.
        from tldw_chatbook.config import get_user_data_dir
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        # One private-SQLite open always starts a helper (ADR-125): proof the
        # HelperLease.start seam still counts, so helper ceilings can't pass
        # at a silent zero.
        connect_private_sqlite(
            "db.base", Path(get_user_data_dir()) / "census_canary.db"
        ).close()

    worker_manager.WorkerManager._new_worker = recording_new_worker
    try:
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
#: timing-dependent, source not isolated -- so its pins are the observed
#: maxima checked with ``OS_OPENS_JITTER_SLACK`` on top; the admission
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
#: 127-133 storage, pinned max + 1; 42,830-44,712 opens). TASK-33642 takes
#: that refresh off the visit path and lowers these again.
#: TASK-33644: one eligible GC interval of the production maintenance loop
#: after it parked, billed on its own (on the census's small database: the
#: graph-epoch read and the retried compaction of the pending collection). Pinned
#: 2026-10-03 at dev 2612fc56b2: 0 / 5 / 2 / 34 in every run (three runs,
#: both evidence variants) -- three owned database calls, two of them on a
#: fresh helper.
MAX_TRACE_GC_PASS_STORAGE_UNITS = {
    "config_admissions": 0,
    "storage_admissions": 5,
    "helper_spawns": 2,
    "os_opens": 34,
}
MAX_VISIT_STORAGE_UNITS = {
    "config_admissions": 43,
    "storage_admissions": 134,
    "helper_spawns": 9,
    "os_opens": 44_712,
}
#: Upward timing jitter allowance on ``os_opens`` only (see above).
OS_OPENS_JITTER_SLACK = 1.05


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
    census = {k: v[0] for k, v in measured.items()}
    request.node.user_properties.append(("storage_units", json.dumps(census)))
    # TASK-33643: every run reports its census -- before any assertion, so a
    # failing run reports too -- and CI ceilings are pinned from those lines;
    # perf-guard.yml names the file and prints it.
    census_log = _census_log_path()
    if census_log is not None:
        with open(census_log, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"case": request.node.name, "platform": sys.platform, "census": census},
                    sort_keys=True,
                )
                + "\n"
            )
    assert (
        counts["settings_readiness_builds"] / KEYSTROKES
        <= MAX_SETTINGS_READINESS_BUILDS_PER_KEY
    )
    for unit in IO_UNITS:
        assert counts[f"canary:{unit}"] >= 1, (
            f"census is blind: the canary (a guarded get_user_data_dir() plus "
            f"one private-SQLite open) counted 0 {unit}; the seam "
            "_count_storage_units wraps for it is no longer on the production "
            "path, so every ceiling below would pass vacuously."
        )
    slack = {"os_opens": OS_OPENS_JITTER_SLACK}
    # The os.open ceilings are macOS measurements (path depth differs per OS),
    # so elsewhere -- the Linux perf-guard runner -- only the logic-level
    # admission and helper counts gate.
    gated = set(IO_UNITS) if sys.platform == "darwin" else set(IO_UNITS) - {"os_opens"}
    over = [
        f"{phase} {unit}: {value} > ceiling {ceilings[unit]}"
        f"{' x ' + str(slack[unit]) if unit in slack else ''}"
        for phase, (values, ceilings) in measured.items()
        for unit, value in values.items()
        if unit in gated and value > ceilings[unit] * slack.get(unit, 1)
    ]
    callers = (
        " Typing-burst callers: " + " | ".join(_TYPING_BURST_CALLERS) + "."
        if any(entry.startswith("typing (whole burst)") for entry in over)
        and _TYPING_BURST_CALLERS
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
