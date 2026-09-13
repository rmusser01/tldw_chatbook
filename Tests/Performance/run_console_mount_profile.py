"""Reproducible warm Console mount profiler for TASK-19505.

Run the production A/B from the repository root::

    .venv/bin/python Tests/Performance/run_console_mount_profile.py \
        --iterations 30 --output /tmp/console-mount-profile.json

Use ``--phase controls`` to reproduce the eager baseline versus empty
Inspector/Context measurement controls. The default production phase
alternates the shipping eager Context rail with the rejected deferred
candidate, injected only inside this measurement process. Every navigation
builds a fresh ``ChatScreen``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTROL_VARIANTS = ("baseline_eager", "inspector_empty_eager", "context_empty")
PRODUCTION_VARIANTS = ("eager", "deferred")


def _configure_isolated_profile(root: Path) -> None:
    home = root / "home"
    data = root / "data"
    config = root / "config"
    for directory in (home, data, config):
        directory.mkdir(parents=True, exist_ok=True)
    config_file = config / "tldw_cli" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n",
        encoding="utf-8",
    )
    os.environ.update(
        {
            "HOME": str(home),
            "XDG_DATA_HOME": str(data),
            "XDG_CONFIG_HOME": str(config),
            "TLDW_CONFIG_PATH": str(config_file),
            "TLDW_TEST_MODE": "1",
            "PYTEST_CURRENT_TEST": "console_mount_profile",
        }
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _outgoing_detached_elapsed_ms(
    unmount_times: dict[str, float], *, started: float
) -> float:
    """Return teardown completion latency without conflating full-ready time."""
    completed_at = unmount_times.get("completed_at")
    if completed_at is None:
        raise RuntimeError("outgoing unmount was not observed")
    return (completed_at - started) * 1000


def _summaries(
    samples: list[dict[str, Any]], variants: tuple[str, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = (
        "first_interactive_ms",
        "full_ready_ms",
        "focus_restore_ms",
        "outgoing_detached_ms",
        "screen_unmount_handler_ms",
        "key_to_echo_ms",
        "enter_to_worker_ms",
    )
    for variant in variants:
        rows = [sample for sample in samples if sample["variant"] == variant]
        metric_summary: dict[str, Any] = {"iterations": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            metric_summary[metric] = {
                "median": round(statistics.median(values), 3),
                "p95": round(_percentile(values, 0.95), 3),
            }
        counts: dict[str, dict[str, float]] = {}
        for count_key in ("first_interactive_widget_counts", "widget_counts"):
            counts = {}
            for name in rows[0][count_key]:
                values = [float(row[count_key][name]) for row in rows]
                counts[name] = {
                    "median": statistics.median(values),
                    "p95": _percentile(values, 0.95),
                }
            metric_summary[count_key] = counts
        result[variant] = metric_summary
    baseline_name = "baseline_eager" if "baseline_eager" in result else "eager"
    baseline = result[baseline_name]
    for variant in variants:
        if variant == baseline_name:
            continue
        candidate = result[variant]
        base = baseline["first_interactive_ms"]["median"]
        measured = candidate["first_interactive_ms"]["median"]
        candidate["first_interactive_improvement_percent"] = round(
            ((base - measured) / base) * 100 if base else 0.0,
            3,
        )
    return result


def _widget_count(widget: Any) -> int:
    return len(list(widget.walk_children(with_self=True)))


def _counts(screen: Any) -> dict[str, int]:
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
    from tldw_chatbook.UI.Console_Modules.right_rail import ConsoleInspectorRail
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    def one(cls: type) -> int:
        matches = list(screen.query(cls))
        return _widget_count(matches[0]) if matches else 0

    def selector(query: str) -> int:
        matches = list(screen.query(query))
        return _widget_count(matches[0]) if matches else 0

    return {
        "screen": _widget_count(screen),
        "context_rail": one(ConsoleLeftRail),
        "transcript": selector("#console-main-column"),
        "inspector_rail": one(ConsoleInspectorRail),
        "composer": one(ConsoleComposerBar),
    }


@contextmanager
def _composition_variant(name: str) -> Iterator[None]:
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
    from tldw_chatbook.UI.Console_Modules.right_rail import ConsoleInspectorRail
    from textual.widgets import Static

    def empty_compose(_self):
        if False:  # pragma: no cover - retain generator semantics
            yield None

    original_left_init = ConsoleLeftRail.__init__
    original_left_compose = ConsoleLeftRail.compose

    def deferred_init(self, *args, **kwargs):
        original_left_init(self, *args, **kwargs)
        self._profile_content_deferred = True

    def deferred_compose(self):
        if getattr(self, "_profile_content_deferred", False):
            placeholder = Static("", id="console-left-rail-deferred", markup=False)
            placeholder.styles.height = 1
            placeholder.styles.min_height = 1
            placeholder.styles.max_height = 1
            yield placeholder
            return
        yield from original_left_compose(self)

    with ExitStack() as stack:
        if name == "deferred":
            stack.enter_context(
                patch.object(ConsoleLeftRail, "__init__", deferred_init)
            )
            stack.enter_context(
                patch.object(ConsoleLeftRail, "compose", deferred_compose)
            )
        if name == "inspector_empty_eager":
            stack.enter_context(
                patch.object(ConsoleInspectorRail, "compose", empty_compose)
            )
        elif name == "context_empty":
            stack.enter_context(patch.object(ConsoleLeftRail, "compose", empty_compose))
        yield


async def _navigate(app: Any, target: str) -> None:
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    await app.handle_screen_navigation(NavigateToScreen(target))


async def _wait_until(predicate, pilot, timeout: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        await pilot.pause(0.001)
    raise RuntimeError("profile condition did not settle")


async def _measure_navigation(
    app: Any,
    pilot: Any,
    *,
    variant: str,
    iteration: int,
) -> dict[str, Any]:
    from textual.screen import Screen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    outgoing = app.screen
    unmount_times: dict[str, float] = {}
    original_unmount = Screen._on_unmount
    original_chat_mount = ChatScreen.on_mount

    def timed_unmount(screen: Screen) -> None:
        if screen is not outgoing:
            original_unmount(screen)
            return
        started = time.perf_counter()
        try:
            original_unmount(screen)
        finally:
            completed_at = time.perf_counter()
            unmount_times["duration_ms"] = (completed_at - started) * 1000
            unmount_times["completed_at"] = completed_at

    started = time.perf_counter()
    first_interactive: float | None = None
    first_interactive_counts: dict[str, int] | None = None
    hydration_completed = asyncio.Event()

    async def timed_hydrate(screen: ChatScreen) -> None:
        nonlocal first_interactive, first_interactive_counts
        if first_interactive is None:
            first_interactive = time.perf_counter()
            first_interactive_counts = _counts(screen)
        try:
            rail = screen.query_one("#console-left-rail")
            if getattr(rail, "_profile_content_deferred", False):
                rail._profile_content_deferred = False
                await rail.recompose()
        finally:
            hydration_completed.set()

    def profiled_chat_mount(screen: ChatScreen) -> None:
        screen.call_after_refresh(screen._profile_mount_boundary)
        original_chat_mount(screen)

    with (
        patch.object(Screen, "_on_unmount", timed_unmount),
        patch.object(
            ChatScreen,
            "_profile_mount_boundary",
            timed_hydrate,
            create=True,
        ),
        patch.object(ChatScreen, "on_mount", profiled_chat_mount),
        _composition_variant(variant),
    ):
        navigation = asyncio.create_task(_navigate(app, "chat"))
        while not navigation.done():
            await pilot.pause(0.001)
            now = time.perf_counter()
            current = app.screen
            if isinstance(current, ChatScreen):
                try:
                    composer = current.query_one(
                        "#console-native-composer", ConsoleComposerBar
                    )
                except Exception:
                    composer = None
                if composer is not None and first_interactive is None:
                    # Fallback for a future screen that no longer schedules
                    # Context hydration at the first-refresh boundary.
                    if (
                        composer.is_mounted
                        and composer.region.width > 0
                        and composer.region.height > 0
                    ):
                        first_interactive = now
                        first_interactive_counts = _counts(current)
        await navigation
        screen = app.screen
        if not isinstance(screen, ChatScreen):
            raise RuntimeError(f"Console did not mount: {type(screen).__name__}")
        await _wait_until(hydration_completed.is_set, pilot)
        await pilot.pause()
    full_ready = time.perf_counter()

    screen = app.screen
    if not isinstance(screen, ChatScreen):
        raise RuntimeError(f"Console did not mount: {type(screen).__name__}")
    composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
    if first_interactive is None:
        first_interactive = full_ready
        first_interactive_counts = _counts(screen)

    await _wait_until(
        lambda: (
            app.focused is composer
            or app.focused in list(composer.walk_children(with_self=True))
        ),
        pilot,
    )
    focus_restored = time.perf_counter()

    before = composer.draft_text()
    echo_started = time.perf_counter()
    await pilot.press("z")
    await _wait_until(lambda: composer.draft_text() != before, pilot)
    key_echo = time.perf_counter()

    composer.load_draft("mount-profile-send")
    worker_started: list[float] = []
    screen._prompt_queue._launch_chain = lambda _draft, _session_id: (
        worker_started.append(time.perf_counter())
    )
    composer.focus()
    await pilot.pause()
    enter_started = time.perf_counter()
    await pilot.press("enter")
    await _wait_until(lambda: bool(worker_started), pilot)

    return {
        "variant": variant,
        "iteration": iteration,
        "first_interactive_ms": (first_interactive - started) * 1000,
        "full_ready_ms": (full_ready - started) * 1000,
        "focus_restore_ms": (focus_restored - started) * 1000,
        "outgoing_detached_ms": _outgoing_detached_elapsed_ms(
            unmount_times,
            started=started,
        ),
        "screen_unmount_handler_ms": unmount_times.get("duration_ms", 0.0),
        "key_to_echo_ms": (key_echo - echo_started) * 1000,
        "enter_to_worker_ms": (worker_started[0] - enter_started) * 1000,
        "first_interactive_widget_counts": first_interactive_counts,
        "widget_counts": _counts(screen),
    }


async def _run(iterations: int, *, phase: str) -> dict[str, Any]:
    from Tests.UI.app_factory import (
        _build_test_app,
        drain_active_service_patches,
        drain_created_dirs,
    )
    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    variants = CONTROL_VARIANTS if phase == "controls" else PRODUCTION_VARIANTS
    samples: list[dict[str, Any]] = []
    try:
        async with app.run_test(size=(170, 48)) as pilot:
            for _ in range(20):
                await pilot.pause(0.05)
            await _navigate(app, "library")
            for iteration in range(iterations):
                offset = iteration % len(variants)
                variant_order = variants[offset:] + variants[:offset]
                for variant in variant_order:
                    sample = await _measure_navigation(
                        app,
                        pilot,
                        variant=variant,
                        iteration=iteration + 1,
                    )
                    samples.append(sample)
                    await _navigate(app, "library")
                    await pilot.pause()
    finally:
        drain_active_service_patches()
        drain_created_dirs()
    return {
        "environment": {
            "iterations_per_variant": iterations,
            "phase": phase,
            "viewport": [170, 48],
            "python": sys.version,
        },
        "summary": _summaries(samples, variants),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--phase",
        choices=("production", "controls"),
        default="production",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    with tempfile.TemporaryDirectory(prefix="tldw-console-mount-profile-") as raw:
        _configure_isolated_profile(Path(raw))
        report = asyncio.run(_run(args.iterations, phase=args.phase))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
