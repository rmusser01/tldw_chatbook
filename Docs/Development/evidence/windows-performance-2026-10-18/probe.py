"""One-off Windows diagnostic: real app, private profile, no provider requests.

Run from repository root with .venv/Scripts/python.exe <this file> --ready.
Add --settings-ab for an alternating, diagnostic-only prepared-summary test.
Add --resume-only for six Home/Console returns; --resume-baseline restores the
pre-attention/workspace modules in memory for a paired original-code control.
Add --full-baseline for the pre-FULL-config-preparation modules, or
--config-observe for diagnostic return-site observations (not timing evidence).
Add --storage-baseline for the pre-initialization-wait storage module.
Add --attach-poll to replay the extra observer used in the earlier stalled runs.
Add --storage-observe for initialization durations and worker stack sampling.
Add --clock-safe to keep probe polling above the Windows loop clock resolution.
Add --presentation-baseline for the pre-CSS/inspector/appearance modules.
Add --matching-baseline for the exact pre-static-filter CSS module.
Add --matching-observe or --matching-benchmark for separate CSS diagnostics.
Add --stalls-baseline for the pre-TASK-35612 presentation modules.
Add --traceback-seconds 1 for a separate accelerated dump stress run (not timing).
Add --profile for cProfile discovery (this Python 3.12 build captures workers
too; its cumulative timings/callers are not trustworthy main-thread evidence).
The separate sampler explicitly selects the main thread. Fresh temporary app
profiles are retained under the original home; delete only those created by
this probe after it exits. No application source or real profile is changed.
"""

import asyncio
from collections import Counter
import cProfile
import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
# Install repository guards before selecting a fresh profile; importing the
# entire conftest would preload application modules before import measurement.
from Tests import network_guard, real_profile_guard  # noqa: E402
from Tests.windows_private_fixture_runner import user_fixture_default_owner  # noqa: E402

fixture_owner = user_fixture_default_owner()
fixture_owner.__enter__()
real_profile_guard.install()
network_guard.install()
scratch = Path(tempfile.mkdtemp(prefix=".tmp-windows-perf-", dir=Path.home()))
for name in ("home", "config", "data"):
    (scratch / name).mkdir()
os.environ.update(
    {
        "HOME": str(scratch / "home"),
        "USERPROFILE": str(scratch / "home"),
        "XDG_CONFIG_HOME": str(scratch / "config"),
        "XDG_DATA_HOME": str(scratch / "data"),
        "TLDW_CONFIG_PATH": str(scratch / "config/config.toml"),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "HF_HUB_OFFLINE": "1",
    }
)

os.environ["TLDW_TEST_MODE"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "windows-performance-diagnostic"
os.environ["TLDW_TEST_CSS_CACHE"] = "0"
Path(os.environ["TLDW_CONFIG_PATH"]).write_text(
    "[first_run]\nsetup_completed = true\n[splash_screen]\nenabled = false\n"
    + (
        '[chat_defaults]\nprovider = "llama_cpp"\nmodel = "local-model"\n'
        '[api_settings.llama_cpp]\napi_url = "http://127.0.0.1:9099"\nmodel = "local-model"\n'
        if "--ready" in sys.argv
        else ""
    ),
    encoding="utf-8",
)
OUT = Path(__file__).parent
PROFILE = "--profile" in sys.argv
TAG = "profiled" if PROFILE else "plain"
if "--ready" in sys.argv:
    TAG += "-ready"
AB = "--settings-ab" in sys.argv
if AB:
    TAG += "-settings-ab"
if "--tag" in sys.argv:
    TAG += "-" + sys.argv[sys.argv.index("--tag") + 1]
stack_log = (OUT / f"{TAG}-stacks.txt").open("w", encoding="utf-8")
# A prior process exited halfway through the C timer's all-thread dump. The
# cause is unproven, but CPython gh-140815 documents races in that GIL-free
# watchdog. Invoke the synchronous dumper from a Python thread instead. This
# cannot diagnose a thread holding the GIL indefinitely (use an external debugger).
stop_tracebacks = threading.Event()
traceback_seconds = (
    float(sys.argv[sys.argv.index("--traceback-seconds") + 1])
    if "--traceback-seconds" in sys.argv
    else 60.0
)
if traceback_seconds <= 0:
    raise ValueError("traceback interval must be positive")
traceback_dumps = 0


def periodic_tracebacks():
    global traceback_dumps
    while not stop_tracebacks.wait(traceback_seconds):
        faulthandler.dump_traceback(file=stack_log)
        traceback_dumps += 1


traceback_thread = threading.Thread(target=periodic_tracebacks, daemon=True)
traceback_thread.start()
report = {
    "python": sys.version,
    "platform": platform.platform(),
    "textual": importlib.metadata.version("textual"),
    "profiled": PROFILE,
    "viewport": [170, 48],
    "routes": [],
    "typing": [],
    "attach_poll": "--attach-poll" in sys.argv,
    "temporary_profile": str(scratch),
    "periodic_tracebacks": "python-thread-synchronous",
    "traceback_interval_seconds": traceback_seconds,
}
stage = "startup"
poll_seconds = 0.005
heartbeat_seconds = 0.01
samples = Counter()
worker_samples = Counter()
stop_sampling = threading.Event()
main_id = threading.get_ident()


def sample_main():
    while not stop_sampling.wait(0.01):
        frame = sys._current_frames().get(main_id)
        stack = []
        while frame is not None:
            stack.append(
                (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
            )
            frame = frame.f_back
        samples[(stage, tuple(stack))] += 1
        if "--storage-observe" in sys.argv:
            for ident, worker_frame in sys._current_frames().items():
                if ident in (main_id, threading.get_ident()):
                    continue
                worker_stack = []
                while worker_frame is not None:
                    worker_stack.append(
                        (
                            worker_frame.f_code.co_filename,
                            worker_frame.f_code.co_name,
                            worker_frame.f_lineno,
                        )
                    )
                    worker_frame = worker_frame.f_back
                worker_samples[(stage, tuple(worker_stack))] += 1


sampler = threading.Thread(target=sample_main, daemon=True)
sampler.start()


def save():
    (OUT / f"{TAG}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


async def until(predicate, seconds=60):
    deadline = time.perf_counter() + seconds
    while not predicate():
        if time.perf_counter() > deadline:
            faulthandler.dump_traceback(file=stack_log)
            raise TimeoutError("diagnostic condition did not arrive")
        await asyncio.sleep(poll_seconds)


async def main():
    global stage, poll_seconds, heartbeat_seconds
    resolution = asyncio.get_running_loop()._clock_resolution
    if "--clock-safe" in sys.argv:
        poll_seconds = max(poll_seconds, 2 * resolution)
        heartbeat_seconds = max(heartbeat_seconds, 2 * resolution)
    report.update(
        loop_clock_resolution_seconds=resolution,
        probe_poll_seconds=poll_seconds,
        probe_heartbeat_seconds=heartbeat_seconds,
    )
    if "--cooperative-timers" in sys.argv:
        from textual import timer

        original_sleep = timer.sleep

        async def coarse_sleep(seconds):
            if seconds >= 0.05:
                await asyncio.sleep(seconds)
            else:
                await original_sleep(seconds)

        timer.sleep = coarse_sleep
        report["cooperative_timers_experiment"] = True
    if "--native-wait-baseline" in sys.argv:
        import native_wait_baseline

        report["baseline_revision"] = native_wait_baseline.BASE_REVISION
        report["baseline_scope"] = ["tldw_chatbook/UI/Screens/home_screen.py"]
    if "--attachment-baseline" in sys.argv:
        import attachment_baseline

        report["baseline_revision"] = attachment_baseline.BASE_REVISION
    if "--return-baseline" in sys.argv:
        import return_baseline

        report["baseline_revision"] = return_baseline.BASE_REVISION
    if "--return-matcher-baseline" in sys.argv:
        import return_matcher_baseline

        report["baseline_revision"] = return_matcher_baseline.BASE_REVISION
        report["baseline_scope"] = sorted(return_matcher_baseline.original.MODULES)
    if "--subject-baseline" in sys.argv:
        import subject_baseline

        report["baseline_revision"] = subject_baseline.BASE_REVISION
    if "--subject-scoped-baseline" in sys.argv:
        import subject_scoped_baseline

        report["baseline_revision"] = subject_scoped_baseline.BASE_REVISION
        report["baseline_scope"] = sorted(subject_scoped_baseline.METHODS)
    if "--stalls-baseline" in sys.argv:
        import stalls_baseline

        report["baseline_revision"] = stalls_baseline.BASE_REVISION
    if "--presentation-baseline" in sys.argv:
        import presentation_baseline

        report["baseline_revision"] = presentation_baseline.BASE_REVISION
    if "--matching-baseline" in sys.argv:
        import matching_baseline

        report["baseline_revision"] = matching_baseline.BASE_REVISION
    if "--storage-baseline" in sys.argv:
        assert "tldw_chatbook.Backup_Recovery.storage_admission" not in sys.modules
        import storage_baseline

        report["baseline_revision"] = storage_baseline.BASE_REVISION
    if "--resume-baseline" in sys.argv:
        import attention_baseline_tests

        report["baseline_revision"] = attention_baseline_tests.BASE_REVISION
    if "--full-baseline" in sys.argv:
        import full_baseline

        report["baseline_revision"] = full_baseline.BASE_REVISION
    profiler = cProfile.Profile()
    if PROFILE:
        profiler.enable()
    start = time.perf_counter()
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from textual import events

    if "--read-handoff" in sys.argv:
        from tldw_chatbook.Chat import console_preparation_reads

        interval = float(sys.argv[sys.argv.index("--read-handoff") + 1])
        console_preparation_reads._READ_HANDOFF_LIMIT = interval
        report["read_handoff_experiment"] = interval

    if "--attachment-observe" in sys.argv:
        from attachment_observer import install

        install(report, lambda: stage)

    finish_config_observation = None
    finish_storage_observation = None
    finish_return_observation = None
    finish_native_wait_observation = None
    if "--native-wait-observe" in sys.argv:
        from native_wait_observer import install

        finish_native_wait_observation = install(report, lambda: stage)
    if "--return-observe" in sys.argv:
        from return_observer import install

        finish_return_observation = install(report, lambda: stage)
    if "--storage-observe" in sys.argv:
        from storage_observer import install

        finish_storage_observation = install(
            OUT / f"{TAG}-initialization.json", lambda: stage
        )
    if "--config-observe" in sys.argv:
        from config_observer import install

        finish_config_observation = install(report, lambda: stage)

    if "--settings-source" in sys.argv:
        import importlib.util

        source = sys.argv[sys.argv.index("--settings-source") + 1]
        name = "tldw_chatbook.UI.Screens.settings_screen"
        original_path = ROOT / "tldw_chatbook/UI/Screens/settings_screen.py"
        spec = importlib.util.spec_from_file_location(name, original_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(
            compile(Path(source).read_bytes(), str(original_path), "exec"),
            module.__dict__,
        )
        report["settings_source"] = source
    report["import_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    app = TldwCli()
    report["construct_ms"] = (time.perf_counter() - start) * 1000
    save()
    try:
        async with app.run_test(size=(170, 48)) as pilot:
            await until(lambda: getattr(app, "_ui_ready", False))
            report["mount_ready_ms"] = (time.perf_counter() - start) * 1000
            await pilot.pause(1)
            report["startup_phases"] = app._startup_phases
            if "--resume-only" in sys.argv:
                app.post_message(NavigateToScreen("chat"))
                await until(lambda: type(app.screen).__name__ == "ChatScreen")
                await pilot.pause(3)
                if "--return-matcher-benchmark" in sys.argv:
                    from subject_benchmark import measure

                    report["subject_benchmark"] = measure(
                        app, "48339e4d6d2fa82b561e6fa1651f8e82928177b2"
                    )
                    save()
                    stage = "shutdown"
                    return
                if "--subject-benchmark" in sys.argv:
                    from subject_benchmark import measure

                    report["subject_benchmark"] = measure(app)
                    save()
                    stage = "shutdown"
                    return
                if "--matching-benchmark" in sys.argv:
                    from matching_benchmark import measure

                    report["matching_benchmark"] = measure(app)
                    save()
                    stage = "shutdown"
                    return
                if "--matching-observe" in sys.argv:
                    from matching_observer import measure

                    report["matching_diagnostic"] = measure(app)
                    save()
                if "--css-observe" in sys.argv:
                    from css_observer import measure

                    report["css_diagnostic"] = measure(app)
                    save()
                report["console_resume"] = []
                for iteration in range(6):
                    stage = "leave-console"
                    app.post_message(NavigateToScreen("home"))
                    await until(lambda: type(app.screen).__name__ == "HomeScreen")
                    await pilot.pause(0.5)
                    stage = f"console-resume-{iteration}"
                    running = True
                    lags = []
                    attach_ms = None
                    resume_screen = None

                    async def heartbeat():
                        nonlocal attach_ms
                        while running:
                            tick = time.perf_counter()
                            await asyncio.sleep(heartbeat_seconds)
                            lags.append(
                                max(0, time.perf_counter() - tick - heartbeat_seconds)
                                * 1000
                            )
                            if (
                                attach_ms is None
                                and resume_screen is not None
                                and resume_screen._console_attach_reconciled
                            ):
                                attach_ms = (time.perf_counter() - started) * 1000

                    beat = asyncio.create_task(heartbeat())
                    await asyncio.sleep(0)
                    started = time.perf_counter()
                    app.post_message(NavigateToScreen("chat"))
                    await until(lambda: type(app.screen).__name__ == "ChatScreen")
                    resume_screen = app.screen
                    composer = app.screen.query_one("#console-native-composer")
                    composer.focus_draft_from(None)
                    before = composer.draft_text()

                    async def attachment_completed():
                        await until(
                            lambda: resume_screen._console_attach_reconciled, 10
                        )
                        return (time.perf_counter() - started) * 1000

                    attach_observer = (
                        asyncio.create_task(attachment_completed())
                        if "--attach-poll" in sys.argv
                        else None
                    )

                    input_started = time.perf_counter()
                    app._driver.send_message(events.Key("z", "z"))
                    try:
                        await until(lambda: composer.draft_text() != before, 10)
                    except TimeoutError:
                        if attach_observer is not None:
                            attach_observer.cancel()
                            await asyncio.gather(
                                attach_observer, return_exceptions=True
                            )
                        faulthandler.dump_traceback(file=stack_log)
                        projection = getattr(
                            app.screen, "_console_readiness_config_projection", None
                        )
                        chains = []
                        for task in asyncio.all_tasks():
                            chain = []
                            coro = task.get_coro()
                            while coro is not None:
                                frame = getattr(coro, "cr_frame", None)
                                if frame is not None:
                                    chain.append(
                                        [
                                            Path(frame.f_code.co_filename).name,
                                            frame.f_code.co_name,
                                            frame.f_lineno,
                                        ]
                                    )
                                coro = getattr(coro, "cr_await", None)
                            if any(
                                "console" in frame[0] or frame[0] == "chat_screen.py"
                                for frame in chain
                            ):
                                chains.append(chain)
                        report["input_timeout"] = {
                            "iteration": iteration,
                            "screen": type(app.screen).__name__,
                            "focused": repr(app.screen.focused),
                            "composer_can_focus": composer.can_focus,
                            "composer_disabled": composer.disabled,
                            "draft_before": before,
                            "draft_after": composer.draft_text(),
                            "attach_reconciled": getattr(
                                app.screen, "_console_attach_reconciled", None
                            ),
                            "sync_in_progress": getattr(
                                app.screen, "_console_sync_in_progress", None
                            ),
                            "workers": [repr(w) for w in app.workers],
                            "projection_pending": getattr(projection, "pending", None),
                            "projection_reads": repr(
                                getattr(projection, "_reads", None)
                            ),
                            "projection_settled": projection._settled.is_set()
                            if projection
                            else None,
                            "tasks": chains,
                        }
                        raise
                    input_ms = (time.perf_counter() - input_started) * 1000
                    arrival_input_ms = (time.perf_counter() - started) * 1000
                    report["pending_resume"] = {
                        "iteration": iteration,
                        "resume_to_input_ms": arrival_input_ms,
                    }
                    save()
                    await pilot.pause(0.5)
                    await until(lambda: app.screen._console_attach_reconciled, 10)
                    if attach_observer is not None:
                        attach_ms = await attach_observer
                    running = False
                    await beat
                    report["console_resume"].append(
                        {
                            "iteration": iteration,
                            "resume_to_input_ms": arrival_input_ms,
                            "resume_to_attach_ms": attach_ms,
                            "input_ms": input_ms,
                            "max_loop_overshoot_ms": max(lags, default=0),
                            "draft_retained": composer.draft_text() == before + "z",
                            "attach_reconciled": app.screen._console_attach_reconciled,
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
                    report.pop("pending_resume", None)
                stage = "shutdown"
                return
            if "--skills-only" in sys.argv:
                app.post_message(NavigateToScreen("chat"))
                await until(lambda: type(app.screen).__name__ == "ChatScreen")
                await pilot.pause(3)
                skill = app.screen._skill
                composer = app.screen.query_one("#console-native-composer")
                composer.focus_draft_from(None)
                await pilot.pause()
                report["skill_context"] = []
                for iteration in range(6):
                    stage = f"skill-context-{iteration}"
                    lags = []
                    running = True

                    async def heartbeat():
                        while running:
                            tick = time.perf_counter()
                            await asyncio.sleep(0.01)
                            lags.append((time.perf_counter() - tick - 0.01) * 1000)

                    heart = asyncio.create_task(heartbeat())
                    await asyncio.sleep(0)
                    before = composer.draft_text()
                    start = time.perf_counter()
                    app._driver.send_message(events.Key("z", "z"))
                    context = await skill._fetch_console_skill_context()
                    elapsed = (time.perf_counter() - start) * 1000
                    await until(lambda: composer.draft_text() != before, 10)
                    input_ms = (time.perf_counter() - start) * 1000
                    running = False
                    await heart
                    report["skill_context"].append(
                        {
                            "iteration": iteration,
                            "elapsed_ms": elapsed,
                            "input_observed_after_fetch_ms": input_ms,
                            "max_loop_overshoot_ms": max(lags, default=0),
                            "available": len(context.get("available_skills", [])),
                            "blocked": len(context.get("blocked_skills", [])),
                        }
                    )
                    save()
                    await pilot.pause(0.1)
                stage = "shutdown"
                return
            prepared_count = None
            if AB or "--warm-settings" in sys.argv:
                from tldw_chatbook.Internal_Prompts.authoring import customized_count
                from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

                prepared_count = await asyncio.to_thread(customized_count)
                report["prepared_prompt_count"] = prepared_count
            if PROFILE:
                profiler.disable()
                profiler.dump_stats(str(OUT / "startup.prof"))
                profiler.clear()
                profiler.enable()
            tour = (
                [("home", "HomeScreen"), ("settings", "SettingsScreen")]
                if AB or "--settings-only" in sys.argv
                else [
                    ("home", "HomeScreen"),
                    ("chat", "ChatScreen"),
                    ("library", "LibraryScreen"),
                    ("settings", "SettingsScreen"),
                ]
            )
            for iteration in range(6 if AB or "--settings-only" in sys.argv else 3):
                for route, expected in tour:
                    stage = f"{route}-{iteration}"
                    variant = "prepared-count" if AB and iteration % 2 else "baseline"
                    replacement = None
                    if variant == "prepared-count" and route == "settings":
                        replacement = patch.object(
                            SettingsScreen,
                            "_get_internal_prompts_customized_count",
                            return_value=prepared_count,
                        )
                        replacement.start()
                    start = time.perf_counter()
                    app.post_message(NavigateToScreen(route))
                    await until(lambda: type(app.screen).__name__ == expected)
                    arrival = time.perf_counter()
                    painted = asyncio.Event()
                    app.screen.call_after_refresh(painted.set)
                    await asyncio.wait_for(painted.wait(), 60)
                    report["routes"].append(
                        {
                            "iteration": iteration,
                            "route": route,
                            "variant": variant,
                            "arrival_ms": (arrival - start) * 1000,
                            "refresh_ms": (time.perf_counter() - start) * 1000,
                            "widgets": len(list(app.screen.walk_children())),
                            "css_sources": len(app.stylesheet.source),
                        }
                    )
                    save()
                    await pilot.pause(0.15)
                    if replacement is not None:
                        replacement.stop()
            stage = "return-console"
            app.post_message(NavigateToScreen("chat"))
            await until(lambda: type(app.screen).__name__ == "ChatScreen")
            await pilot.pause(0.5)
            composer = app.screen.query_one("#console-native-composer")
            composer.focus_draft_from(None)
            await pilot.pause()
            stage = "typing"
            for _ in range(12):
                before = composer.draft_text()
                start = time.perf_counter()
                app._driver.send_message(events.Key("z", "z"))
                await until(lambda: composer.draft_text() != before, 10)
                report["typing"].append((time.perf_counter() - start) * 1000)
            # Idle loop overshoot: long stalls distinguish blocking work from paint cost.
            lags = []
            stage = "idle"
            cpu = time.process_time()
            idle_start = time.perf_counter()
            while time.perf_counter() - idle_start < 3:
                tick = time.perf_counter()
                await asyncio.sleep(0.02)
                lags.append(max(0, time.perf_counter() - tick - 0.02) * 1000)
            report["idle"] = {
                "cpu_seconds": time.process_time() - cpu,
                "elapsed_seconds": time.perf_counter() - idle_start,
                "loop_overshoot_ms": lags,
            }
            stage = "display-verification"
            projection = getattr(
                app.screen, "_console_readiness_config_projection", None
            )
            estimate = (
                app.screen._context_spend._active_console_settings_context_estimate()
            )
            report["tokenizer_display"] = {
                "prepared": getattr(projection, "_tokenizer_prepared", None)
                is not None,
                "used_tokens": estimate.used_tokens,
                "label": estimate.label,
            }
            stage = "shutdown"
    except BaseException as error:
        report["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        faulthandler.dump_traceback(file=stack_log)
        raise
    finally:
        if finish_native_wait_observation is not None:
            finish_native_wait_observation()
        if finish_return_observation is not None:
            finish_return_observation()
        if finish_storage_observation is not None:
            finish_storage_observation()
        if finish_config_observation is not None:
            try:
                finish_config_observation()
            except Exception as error:
                report["config_observer_error"] = repr(error)
        stop_sampling.set()
        sampler.join(1)
        if "--storage-observe" in sys.argv:
            (OUT / f"{TAG}-worker-samples.json").write_text(
                json.dumps(
                    [
                        {"stage": name, "stack": stack, "count": count}
                        for (name, stack), count in worker_samples.most_common()
                    ]
                ),
                encoding="utf-8",
            )
        (OUT / f"{TAG}-samples.json").write_text(
            json.dumps(
                [
                    {"stage": name, "stack": stack, "count": count}
                    for (name, stack), count in samples.most_common()
                ]
            ),
            encoding="utf-8",
        )
        report["final_screen"] = type(app.screen).__name__ if app.screen_stack else None
        if PROFILE:
            profiler.disable()
            profiler.dump_stats(str(OUT / "main-thread.prof"))
            with (OUT / "hotspots.txt").open("w", encoding="utf-8") as handle:
                stats = pstats.Stats(profiler, stream=handle)
                stats.sort_stats("cumulative").print_stats(65)
                stats.sort_stats("tottime").print_stats(50)
        report["blocked_network_attempts"] = len(network_guard.blocked_attempts())
        report["real_profile_refusals"] = len(real_profile_guard.take_violations())
        report["periodic_traceback_dumps"] = traceback_dumps
        save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        stop_tracebacks.set()
        traceback_thread.join()
        stack_log.close()
        fixture_owner.__exit__(None, None, None)
