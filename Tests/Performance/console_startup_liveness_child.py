"""One original App/Pilot startup inside a selected private interpreter.

The driver owns environment/process/source selection. This liveness child keeps
original App behavior and deadlines; it installs no I/O profiler or syscall audit.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time


async def observe(args, result):
    from Tests import network_guard, real_profile_guard

    stages = result["stages"] = []
    intervals = result["heartbeat_intervals"] = []
    census = None
    timing = None
    current = None
    stop = False

    def stage(name):
        nonlocal current
        now = time.perf_counter()
        if current is not None:
            current["exited"] = now
        current = {"stage": name, "entered": now, "exited": None}
        stages.append(current)
        if census is not None:
            census.phase = name
        return now

    async def heartbeat():
        previous = time.perf_counter()
        try:
            while not stop:
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                intervals.append(
                    {
                        "entered": previous,
                        "exited": now,
                        "gap_seconds": max(0, now - previous - 0.02),
                    }
                )
                previous = now
        finally:
            now = time.perf_counter()
            intervals.append(
                {
                    "entered": previous,
                    "exited": now,
                    "gap_seconds": max(0, now - previous - 0.02),
                }
            )

    heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # Arm heartbeat before synchronous import/construction.
    try:
        stage("python_import")
        from tldw_chatbook.app import TldwCli

        if args.mode == "timing":
            setup_entered = time.perf_counter()
            timing_path = Path(__file__).with_name("console_startup_timing_witness.py")
            assert args.expected_timing_sha256 is not None
            assert (
                hashlib.sha256(timing_path.read_bytes()).hexdigest()
                == args.expected_timing_sha256
            )
            from Tests.Performance.console_startup_timing_witness import (
                StartupTimingWitness,
            )

            result["executed_probe_helpers"]["timing"] = {
                "path": str(timing_path.resolve()),
                "raw_sha256": args.expected_timing_sha256,
                "start_callable": result["_source_observer"](
                    StartupTimingWitness.start, args.repo
                ),
            }
            timing = StartupTimingWitness(args.repo)
            try:
                timing.start()
            finally:
                setup_exited = time.perf_counter()
                result["timing_setup"] = {
                    "entered": setup_entered,
                    "exited": setup_exited,
                    "seconds": setup_exited - setup_entered,
                    "coverage": "Named diagnostic setup within import-stage envelope; outside original constructor interval.",
                }

        stage("app_constructor")
        app = TldwCli()
        stage("mount_until_first_input")
        async with app.run_test(size=(140, 42)) as pilot:
            # Initial receipt I/O may outlive run_test entry. Keep waiting
            # inside the unchanged measured mount stage and 240s bound.
            while not getattr(app, "_initial_screen_pushed", False):
                await asyncio.sleep(0.01)
            screen = app.screen
            assert type(screen).__name__ == "ChatScreen"
            composer = screen._console_composer_or_none()
            assert composer is not None, "Composer unavailable: no usable-input witness"
            original_key = type(screen).on_key
            original_insert = type(composer).insert_text
            original_dispatch = type(composer).handle_console_key
            helper = result.pop("_source_observer")
            result["original_key_callback_sources"] = {
                "screen_key": helper(original_key, args.repo),
                "composer_insert": helper(original_insert, args.repo),
                "composer_dispatch": helper(original_dispatch, args.repo),
            }
            before = composer.draft_text()
            if census is not None:
                census.expected_screen, census.expected_composer = screen, composer
            composer.focus()
            caret = composer._cursor_index
            assert (
                not composer._draft_selection_all
                and composer._draft_selection_range is None
            )
            assert 0 <= caret <= len(before)
            result["key_posted"] = time.perf_counter()
            await pilot.press("k")  # Actual driver delivery; no direct draft insert.
            accepted = time.perf_counter()
            after = composer.draft_text()
            assert after == before[:caret] + "k" + before[caret:]
            assert type(screen).on_key is original_key
            assert type(composer).insert_text is original_insert
            assert type(composer).handle_console_key is original_dispatch
            result["usable_input"] = {
                "accepted": accepted,
                "seconds_from_child_entry": accepted - result["child_entered"],
                "seconds_from_parent_spawn": accepted - args.parent_spawn,
                "seconds_from_constructor_entry": accepted - stages[1]["entered"],
                "screen_actor_id": id(screen),
                "composer_actor_id": id(composer),
                "draft_length_delta": len(after) - len(before),
                "delivery": "Original Pilot.press -> normal key/composer event path",
                "original_callbacks_retained": True,
            }
            stage("usable_settle")
            await asyncio.sleep(3)
            result["mounted_heartbeat_window_end"] = time.perf_counter()
            if census is not None:
                assert {row["source"] for row in census.key_events} >= {
                    "screen_key",
                    "composer_key_dispatch",
                    "composer_insert",
                }, "The actual normal key/composer route was not observed"
                result["io_census"] = census.stop()
                assert result["io_census"]["profile_owned_and_restored"]
                assert result["io_census"]["original_callable_identities_unchanged"]
                assert not result["io_census"]["source_mismatches"]
                census = None
            # Fresh durable state inspection is outside the measured startup I/O
            # window. It does not substitute for actual read/activation events.
            from tldw_chatbook.DB.VisualIdentity_DB import LOCAL_OWNER_ID

            with app.chachanotes_db.transaction() as cursor:
                result["durable_builtin_state"] = {
                    "owner_user_id": LOCAL_OWNER_ID,
                    "pack_count": cursor.execute(
                        "SELECT COUNT(*) FROM visual_identity_packs WHERE owner_user_id = ? AND source_kind = ?",
                        (LOCAL_OWNER_ID, "builtin"),
                    ).fetchone()[0],
                }
            stage("shutdown")
        result["original_app_shutdown_completed"] = True
        stage("after_shutdown")
        result["complete"] = True
    finally:
        timing_stop_failure = None
        timing_primary_error_present = sys.exception() is not None
        if timing is not None:
            try:
                result["startup_timing"] = timing.stop()
            except BaseException as error:
                timing_stop_failure = type(error).__name__
                result["startup_timing_retirement_error_type"] = timing_stop_failure
        if census is not None:
            result["io_census"] = census.stop()
        stop = True
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        if current is not None:
            current["exited"] = time.perf_counter()
        result["heartbeat_by_stage"] = {
            row["stage"]: {
                "intervals": sum(
                    item["entered"] < row["exited"] and item["exited"] > row["entered"]
                    for item in intervals
                ),
                "max_gap_seconds": max(
                    (
                        item["gap_seconds"]
                        for item in intervals
                        if item["entered"] < row["exited"]
                        and item["exited"] > row["entered"]
                    ),
                    default=0,
                ),
                "over_100ms": sum(
                    item["gap_seconds"] > 0.1
                    for item in intervals
                    if item["entered"] < row["exited"]
                    and item["exited"] > row["entered"]
                ),
            }
            for row in stages
            if row["exited"] is not None
        }
        result["network_attempts"] = len(network_guard.blocked_attempts())
        result["real_profile_guard_refusals"] = len(
            real_profile_guard.take_violations()
        )
        assert result["network_attempts"] == result["real_profile_guard_refusals"] == 0
        if timing_stop_failure is not None and not timing_primary_error_present:
            raise RuntimeError(
                "Startup timing observer retirement failed: " + timing_stop_failure
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--mode", choices=("liveness", "timing"), required=True)
    parser.add_argument("--launch", choices=("cold", "warm"), required=True)
    parser.add_argument("--parent-spawn", type=float, required=True)
    parser.add_argument("--driver-pid", type=int, required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--expected-child-sha256", required=True)
    parser.add_argument("--expected-timing-sha256")
    args = parser.parse_args()
    args.repo = args.repo.absolute()
    entered = time.perf_counter()
    assert not args.receipt.exists(), "Never overwrite a prior startup receipt"
    # This evidence helper has only stdlib top-level imports and does not run
    # its parent's main entry point, import the app, or replace any app source.
    driver_path = Path(__file__).with_name("run_console_startup_liveness.py")
    assert (
        hashlib.sha256(driver_path.read_bytes()).hexdigest()
        == args.expected_driver_sha256
    )
    assert (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == args.expected_child_sha256
    )
    process_helpers = runpy.run_path(
        str(driver_path), run_name="startup_process_identity_helpers"
    )
    process_identity = process_helpers["child_process_receipt"](args.driver_pid)
    process_helpers["configure_managed_source_roots"](args.repo)
    sys.path.insert(0, str(Path(__file__).parent))
    # Environment is selected by the parent before importing app/config sources.
    selector = Path(os.environ["TLDW_CONFIG_PATH"])
    profile_root = Path(os.environ["TLDW_STARTUP_PROFILE_ROOT"])
    assert selector.is_relative_to(profile_root) and selector.is_file()
    from Tests import network_guard, real_profile_guard
    from Tests.windows_private_fixture_runner import user_fixture_default_owner

    real_profile_guard.install()
    network_guard.install()
    result = {
        "complete": False,
        "diagnostic_only": args.mode == "io",
        "pid": os.getpid(),
        "child_entered": entered,
        "mode": args.mode,
        "process_identity": process_identity,
        "executed_probe_helpers": {
            "driver": {
                "path": str(driver_path.resolve()),
                "raw_sha256": args.expected_driver_sha256,
            },
            "child": {
                "path": str(Path(__file__).resolve()),
                "raw_sha256": args.expected_child_sha256,
            },
            "driver_helper_actual_code": {
                name: str(Path(process_helpers[name].__code__.co_filename).resolve())
                for name in (
                    "child_process_receipt",
                    "configure_managed_source_roots",
                    "loaded_source_receipt",
                    "observed_callable_source",
                )
            },
        },
        "launch": args.launch,
        "platform": platform.platform(),
        "python": sys.version,
        "parent_spawn": args.parent_spawn,
        "selected_config_sha256_before": hashlib.sha256(
            selector.read_bytes()
        ).hexdigest(),
        "limits": "Fresh process/app profile, original Pilot default wait/deadlines, "
        "original timers/workers/seeders/guards. OS page cache uncontrolled. "
        "Global profiled time cannot qualify performance budget success.",
    }
    if args.mode == "timing":
        result.update(
            diagnostic_only=True, budget_acceptance_eligible=False, budgets_pass=None
        )
    result["_source_observer"] = process_helpers["observed_callable_source"]
    try:
        with user_fixture_default_owner():

            async def bounded_app():
                # Retain the existing whole-App bound and await original cleanup.
                return await asyncio.wait_for(observe(args, result), timeout=240)

            asyncio.run(bounded_app())
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.pop("_source_observer", None)
        source_failure = None
        try:
            loaded, origins = process_helpers["loaded_source_receipt"](args.repo)
            result["loaded_source_sha256"] = loaded
            result["actual_loaded_module_origins"] = origins
            assert (
                hashlib.sha256(driver_path.read_bytes()).hexdigest()
                == args.expected_driver_sha256
            )
            assert (
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                == args.expected_child_sha256
            )
            result["source_receipt_accepted"] = True
        except BaseException as error:
            source_failure = error
            result["source_receipt_accepted"] = False
            result["source_receipt_error"] = {
                "type": type(error).__name__,
                "reason": str(error),
            }
        result["selected_config_sha256_after"] = hashlib.sha256(
            selector.read_bytes()
        ).hexdigest()
        result["exited_after_original_shutdown"] = time.perf_counter()
        args.receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if source_failure is not None:
            raise source_failure


if __name__ == "__main__":
    main()
