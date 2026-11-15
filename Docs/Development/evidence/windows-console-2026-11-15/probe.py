"""Native guarded Console return probe; plain and observed runs stay separate."""

import asyncio
from collections import Counter
import argparse
import cProfile
import hashlib
import subprocess
import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(
    description="Guarded native six-return Console journey"
)
for flag in (
    "ready",
    "resume-only",
    "clock-safe",
    "attach-poll",
    "return-observe",
    "cpu-profile",
):
    parser.add_argument("--" + flag, action="store_true")
parser.add_argument("--tag", default="current")
args = parser.parse_args()
if not args.resume_only:
    parser.error("--resume-only is required")
if not args.tag or any(
    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    for c in args.tag
):
    parser.error("unsafe output tag")

sys.path.insert(0, str(ROOT))
# Install repository guards before selecting a fresh profile; importing the
# entire conftest would preload application modules before import measurement.
from Tests import network_guard, real_profile_guard  # noqa: E402
from Tests.windows_private_fixture_runner import user_fixture_default_owner  # noqa: E402

# Keep native token restoration independent of diagnostics and report IO.
with user_fixture_default_owner():
    real_profile_guard.install()
    network_guard.install()
    # A denied Windows mkdtemp can retry thousands of candidates when the
    # sandbox disagrees with os.access. One unique mkdir fails immediately.
    scratch = Path.home() / (".tmp-windows-perf-" + uuid.uuid4().hex[:12])
    scratch.mkdir(mode=0o700)
    for name in ("home", "config", "data", "temp"):
        (scratch / name).mkdir()
    os.environ.update(
        {
            "HOME": str(scratch / "home"),
            "USERPROFILE": str(scratch / "home"),
            "TEMP": str(scratch / "temp"),
            "TMP": str(scratch / "temp"),
            "XDG_CONFIG_HOME": str(scratch / "config"),
            "XDG_DATA_HOME": str(scratch / "data"),
            "TLDW_CONFIG_PATH": str(scratch / "config/config.toml"),
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "HF_HUB_OFFLINE": "1",
        }
    )

    tempfile.tempdir = None
    os.environ["TLDW_TEST_MODE"] = "1"
    os.environ["PYTEST_CURRENT_TEST"] = "windows-performance-diagnostic"
    os.environ["TLDW_TEST_CSS_CACHE"] = "0"
    Path(os.environ["TLDW_CONFIG_PATH"]).write_text(
        "[first_run]\nsetup_completed = true\n[splash_screen]\nenabled = false\n"
        + (
            '[chat_defaults]\nprovider = "llama_cpp"\nmodel = "local-model"\n'
            '[api_settings.llama_cpp]\napi_url = "http://127.0.0.1:9099"\nmodel = "local-model"\n'
            if args.ready
            else ""
        ),
        encoding="utf-8",
    )
    OUT = Path(__file__).parent
    TAG = ("diagnostic" if args.return_observe else "plain") + "-" + args.tag
    stack_log = (OUT / f"{TAG}-stacks.txt").open("w", encoding="utf-8")
    # A prior process exited halfway through the C timer's all-thread dump. The
    # cause is unproven, but CPython gh-140815 documents races in that GIL-free
    # watchdog. Invoke the synchronous dumper from a Python thread instead. This
    # cannot diagnose a thread holding the GIL indefinitely (use an external debugger).
    stop_tracebacks = threading.Event()
    traceback_seconds = 60.0
    traceback_dumps = 0

    def periodic_tracebacks():
        global traceback_dumps
        while not stop_tracebacks.wait(traceback_seconds):
            faulthandler.dump_traceback(file=stack_log)
            traceback_dumps += 1

    traceback_thread = threading.Thread(target=periodic_tracebacks, daemon=True)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "textual": importlib.metadata.version("textual"),
        "diagnostic": args.return_observe,
        "viewport": [170, 48],
        "routes": [],
        "typing": [],
        "attach_poll": args.attach_poll,
        "temporary_profile": str(scratch),
        "periodic_tracebacks": "python-thread-synchronous",
        "traceback_interval_seconds": traceback_seconds,
    }
    stage = "startup"
    poll_seconds = 0.005
    heartbeat_seconds = 0.01
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

    def save():
        (OUT / f"{TAG}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    async def until(predicate, seconds=60):
        deadline = time.perf_counter() + seconds
        while not predicate():
            if time.perf_counter() > deadline:
                faulthandler.dump_traceback(file=stack_log)
                raise TimeoutError("diagnostic condition did not arrive")
            await asyncio.sleep(poll_seconds)

    def product_sources():
        return {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "tldw_chatbook").rglob("*.py"))
        }

    def write_source_manifest(label, sources):
        filename = f"{TAG}-sources-{label}.json"
        data = (json.dumps(sources, sort_keys=True, indent=2) + "\n").encode("utf-8")
        (OUT / filename).write_bytes(data)
        return {
            "file": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "count": len(sources),
        }

    def revision():
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()

    async def resume_once(iteration, app, pilot, navigate_type, events):
        global stage
        stage = "leave-console"
        app.post_message(navigate_type("home"))
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
                    max(0, time.perf_counter() - tick - heartbeat_seconds) * 1000
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
        cpu_profile = (
            cProfile.Profile() if args.cpu_profile and iteration == 0 else None
        )
        if cpu_profile is not None:
            cpu_profile.enable()
        if args.return_observe:
            report.setdefault("return_journey_edges", []).append(
                [stage, "navigate", started]
            )
        app.post_message(navigate_type("chat"))
        await until(lambda: type(app.screen).__name__ == "ChatScreen")
        resume_screen = app.screen
        composer = app.screen.query_one("#console-native-composer")
        composer.focus_draft_from(None)
        before = composer.draft_text()

        async def attachment_completed():
            await until(lambda: resume_screen._console_attach_reconciled, 10)
            return (time.perf_counter() - started) * 1000

        attach_observer = (
            asyncio.create_task(attachment_completed()) if args.attach_poll else None
        )

        input_started = time.perf_counter()
        if args.return_observe:
            report["return_journey_edges"].append([stage, "key-posted", input_started])
        app._driver.send_message(events.Key("z", "z"))
        await until(lambda: composer.draft_text() != before, 10)
        input_ms = (time.perf_counter() - input_started) * 1000
        arrival_input_ms = (time.perf_counter() - started) * 1000
        if args.return_observe:
            report["return_journey_edges"].append(
                [stage, "key-observed", time.perf_counter()]
            )
        report["pending_resume"] = {
            "iteration": iteration,
            "resume_to_input_ms": arrival_input_ms,
        }
        save()
        await pilot.pause(0.5)
        await until(lambda: app.screen._console_attach_reconciled, 10)
        if attach_observer is not None:
            attach_ms = await attach_observer
        if args.return_observe:
            report["return_journey_edges"].append(
                [stage, "attach-poll-returned", time.perf_counter()]
            )
        running = False
        await beat
        if cpu_profile is not None:
            cpu_profile.disable()
            cpu_profile.dump_stats(str(OUT / f"{TAG}-first-return.pstats"))
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
        assert (
            composer.draft_text() == before + "z"
        ), "posted draft key was not retained"
        assert app.screen._console_attach_reconciled, "attachment incomplete"

    async def main():
        global stage, poll_seconds, heartbeat_seconds
        resolution = asyncio.get_running_loop()._clock_resolution
        if args.clock_safe:
            poll_seconds = max(poll_seconds, 2 * resolution)
            heartbeat_seconds = max(heartbeat_seconds, 2 * resolution)
        report.update(
            loop_clock_resolution_seconds=resolution,
            probe_poll_seconds=poll_seconds,
            probe_heartbeat_seconds=heartbeat_seconds,
        )
        start = time.perf_counter()
        from tldw_chatbook.app import TldwCli
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
        from textual import events

        finish = None
        try:
            if args.return_observe:
                from return_observer import install

                finish = install(report, lambda: stage)
            report["import_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            app = TldwCli()
            report["construct_ms"] = (time.perf_counter() - start) * 1000
            save()
            async with app.run_test(size=(170, 48)) as pilot:
                await until(lambda: getattr(app, "_ui_ready", False))
                report["mount_ready_ms"] = (time.perf_counter() - start) * 1000
                await pilot.pause(1)
                app.post_message(NavigateToScreen("chat"))
                await until(lambda: type(app.screen).__name__ == "ChatScreen")
                await pilot.pause(3)
                # Qualify a settled starting visit before measuring returns.
                # An unfinished initial attach is a startup failure, not a
                # return-latency sample. Keep the measured 10s bound unchanged.
                await until(lambda: app.screen._console_attach_reconciled)
                report["console_resume"] = []
                for iteration in range(6):
                    await resume_once(iteration, app, pilot, NavigateToScreen, events)
        except BaseException as error:
            report["failure"] = {"stage": stage, "type": type(error).__name__}
            raise
        finally:
            stage = "shutdown"
            if finish is not None:
                finish()

    if __name__ == "__main__":
        before = None
        head = None
        try:
            before = product_sources()
            head = revision()
            report.update(
                revision=head,
                product_sources_before=write_source_manifest("before", before),
            )
            traceback_thread.start()
            sampler.start()
            asyncio.run(main())
        finally:
            stop_sampling.set()
            stop_tracebacks.set()
            if sampler.ident is not None:
                sampler.join(2)
            if traceback_thread.ident is not None:
                traceback_thread.join(2)
            report["observer_threads_stopped"] = (
                not sampler.is_alive() and not traceback_thread.is_alive()
            )
            stack_log.close()
            report["blocked_network_attempts"] = len(network_guard.blocked_attempts())
            report["real_profile_refusals"] = len(real_profile_guard.take_violations())
            after = product_sources()
            report["product_sources_after"] = write_source_manifest("after", after)
            report["revision_after"] = revision()
            report["product_sources_unchanged"] = (
                None if before is None else after == before
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
            save()
            if before is not None:
                assert report["product_sources_unchanged"], "product source changed"
            if head is not None:
                assert report["revision_after"] == head, "HEAD changed"
            assert report["observer_threads_stopped"], "observer teardown incomplete"
            assert (
                report["blocked_network_attempts"]
                == report["real_profile_refusals"]
                == 0
            ), "guard violation"
