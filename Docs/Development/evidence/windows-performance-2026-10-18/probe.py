"""One-off Windows diagnostic: real app, private profile, no provider requests.

Run from repository root with .venv/Scripts/python.exe <this file> --ready.
Add --settings-ab for an alternating, diagnostic-only prepared-summary test.
Add --resume-only for six Home/Console returns; --resume-baseline restores the
pre-attention/workspace modules in memory for a paired original-code control.
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
faulthandler.dump_traceback_later(60, repeat=True, file=stack_log)
report = {
    "python": sys.version,
    "platform": platform.platform(),
    "textual": importlib.metadata.version("textual"),
    "profiled": PROFILE,
    "viewport": [170, 48],
    "routes": [],
    "typing": [],
    "temporary_profile": str(scratch),
}
stage = "startup"
samples = Counter()
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


sampler = threading.Thread(target=sample_main, daemon=True)
sampler.start()


def save():
    (OUT / f"{TAG}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


async def until(predicate, seconds=60):
    deadline = time.perf_counter() + seconds
    while not predicate():
        if time.perf_counter() > deadline:
            raise TimeoutError("diagnostic condition did not arrive")
        await asyncio.sleep(0.005)


async def main():
    global stage
    if "--resume-baseline" in sys.argv:
        import attention_baseline_tests

        report["baseline_revision"] = attention_baseline_tests.BASE_REVISION
    profiler = cProfile.Profile()
    if PROFILE:
        profiler.enable()
    start = time.perf_counter()
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from textual import events

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
                report["console_resume"] = []
                for iteration in range(6):
                    stage = "leave-console"
                    app.post_message(NavigateToScreen("home"))
                    await until(lambda: type(app.screen).__name__ == "HomeScreen")
                    await pilot.pause(.5)
                    stage = f"console-resume-{iteration}"
                    running = True
                    lags = []

                    async def heartbeat():
                        while running:
                            tick = time.perf_counter()
                            await asyncio.sleep(0.01)
                            lags.append(
                                max(0, time.perf_counter() - tick - 0.01) * 1000
                            )

                    beat = asyncio.create_task(heartbeat())
                    await asyncio.sleep(0)
                    started = time.perf_counter()
                    app.post_message(NavigateToScreen("chat"))
                    await until(lambda: type(app.screen).__name__ == "ChatScreen")
                    composer = app.screen.query_one("#console-native-composer")
                    composer.focus_draft_from(None)
                    before = composer.draft_text()
                    input_started = time.perf_counter()
                    app._driver.send_message(events.Key("z", "z"))
                    await until(lambda: composer.draft_text() != before, 10)
                    input_ms = (time.perf_counter() - input_started) * 1000
                    arrival_input_ms = (time.perf_counter() - started) * 1000
                    await pilot.pause(0.5)
                    await until(lambda: app.screen._console_attach_reconciled, 10)
                    running = False
                    await beat
                    report["console_resume"].append(
                        {
                            "iteration": iteration,
                            "resume_to_input_ms": arrival_input_ms,
                            "input_ms": input_ms,
                            "max_loop_overshoot_ms": max(lags, default=0),
                            "draft_retained": composer.draft_text() == before + "z",
                            "attach_reconciled": app.screen._console_attach_reconciled,
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
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
                    report["skill_context"].append({
                        "iteration": iteration, "elapsed_ms": elapsed,
                        "input_observed_after_fetch_ms": input_ms,
                        "max_loop_overshoot_ms": max(lags, default=0),
                        "available": len(context.get("available_skills", [])),
                        "blocked": len(context.get("blocked_skills", [])),
                    })
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
            projection = getattr(app.screen, "_console_readiness_config_projection", None)
            estimate = app.screen._context_spend._active_console_settings_context_estimate()
            report["tokenizer_display"] = {
                "prepared": getattr(projection, "_tokenizer_prepared", None) is not None,
                "used_tokens": estimate.used_tokens,
                "label": estimate.label,
            }
            stage = "shutdown"
    finally:
        stop_sampling.set()
        sampler.join(1)
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
        save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        faulthandler.cancel_dump_traceback_later()
        stack_log.close()
        fixture_owner.__exit__(None, None, None)
