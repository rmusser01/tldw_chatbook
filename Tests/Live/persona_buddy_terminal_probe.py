#!/usr/bin/env python3
"""Bounded POSIX-PTY verification for the native Persona Buddy interactions."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from typing import Any

_TIMEOUT_SECONDS = 12.0


def _child(preferences_path: Path, report_path: Path) -> int:
    """Run the production-CSS child application inside the allocated PTY."""

    from dataclasses import asdict, replace

    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.screen import ModalScreen
    from textual.widgets import Input, Static

    from tldw_chatbook.Persona_Buddy import (
        PersonaBuddyController,
        PersonaBuddyPreferences,
        PersonaBuddySelection,
        parse_persona_buddy_preferences,
        serialize_persona_buddy_preferences,
    )
    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
    from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
        PersonaBuddyWidget,
    )
    from tldw_chatbook.css import build_css

    if preferences_path.exists():
        raw = json.loads(preferences_path.read_text(encoding="utf-8"))
        preferences = parse_persona_buddy_preferences(raw)
    else:
        preferences = PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "terminal-probe"),
        )
    loaded_geometry = asdict(preferences.geometry)

    def write_preferences(value: PersonaBuddyPreferences) -> bool:
        temporary = preferences_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(serialize_persona_buddy_preferences(value), sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(preferences_path)
        return True

    class ProbeScreen(BaseAppScreen):
        def __init__(self, app_instance: "ProbeApp", name: str) -> None:
            super().__init__(app_instance, name)

        def compose_content(self) -> ComposeResult:
            yield Input(id="terminal-focus-probe")
            yield Static(f"SCREEN {self.screen_name}", id="terminal-screen-label")

    class ModalHitSurface(Static, can_focus=True):
        DEFAULT_CSS = "ModalHitSurface { width: 100%; height: 100%; }"

        def on_mouse_down(self, event: events.MouseDown) -> None:
            self.app.modal_hits += 1
            self.app.run_worker(
                self.app.action_probe_navigate(),
                group="terminal-probe-navigation",
                exclusive=True,
            )
            event.stop()

    class BlockingModal(ModalScreen):
        BINDINGS = [
            Binding("escape", "close_probe_modal", "Close", priority=True),
            Binding("d", "close_probe_modal", "Close", priority=True),
        ]

        def compose(self) -> ComposeResult:
            yield ModalHitSurface("MODAL BLOCKER", id="terminal-modal-blocker")

        def action_close_probe_modal(self) -> None:
            self.dismiss()

    css_dir = Path(build_css.__file__).parent
    screen_scoped, screen_self = build_css.screen_css_paths(css_dir)

    class ProbeApp(App):
        CSS_PATH = [
            str(screen_scoped),
            str(css_dir / "tldw_cli_modular.tcss"),
            str(screen_self),
        ]
        BINDINGS = [
            Binding("m", "probe_modal", "Modal", priority=True),
            Binding("n", "probe_navigate", "Navigate", priority=True),
            Binding("o", "probe_reopen", "Reopen", priority=True),
            Binding("q", "probe_finish", "Finish", priority=True),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.persona_buddy_controller = PersonaBuddyController(
                preferences=preferences,
                preference_writer=write_preferences,
            )

            async def keep_probe_visual_unknown(*, cols: int, lines: int):
                return None

            self.persona_buddy_controller.resolve_current_visual = (
                keep_probe_visual_unknown
            )
            self.modal_hits = 0
            self.navigation_count = 0
            self.initial_geometry = loaded_geometry
            self.focus_guard_observed = False
            self.modal_timer_fired = False

        def _get_default_css(self):  # noqa: D102 - mirrors production CSS loading
            return (
                build_css.widget_defaults_sources(css_dir) + super()._get_default_css()
            )

        async def reconcile_persona_buddy_view(self) -> None:
            screen = self.screen
            if isinstance(screen, BaseAppScreen) and screen.is_active:
                await screen.reconcile_persona_buddy_view()

        async def on_mount(self) -> None:
            await self.push_screen(ProbeScreen(self, "first"))
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGUSR1,
                self._write_probe_report,
            )
            self.call_after_refresh(self._capture_focus_guard)
            self.set_interval(0.10, self._write_probe_report)

        def _capture_focus_guard(self) -> None:
            screen = self.screen
            buddies = list(screen.query(PersonaBuddyWidget))
            if not isinstance(screen, ProbeScreen) or not buddies:
                self.call_after_refresh(self._capture_focus_guard)
                return
            focus_probe = screen.query_one("#terminal-focus-probe", Input)
            screen.set_focus(focus_probe, scroll_visible=False)
            buddies[0].refresh_from_controller()
            self.focus_guard_observed = screen.focused is focus_probe

        async def action_probe_modal(self) -> None:
            self.modal_timer_fired = True
            await self.push_screen(BlockingModal())

        async def action_probe_navigate(self) -> None:
            self.navigation_count += 1
            await self.switch_screen(ProbeScreen(self, "second"))

        async def action_probe_reopen(self) -> None:
            current = self.persona_buddy_controller.current_preferences()
            await self.persona_buddy_controller.update_preferences(
                replace(current, open=True)
            )
            await self.reconcile_persona_buddy_view()

        def action_probe_finish(self) -> None:
            self._write_probe_report()

        def _write_probe_report(self) -> None:
            screen = next(
                screen
                for screen in reversed(tuple(self.screen_stack))
                if isinstance(screen, ProbeScreen)
            )
            buddies = list(screen.query(PersonaBuddyWidget))
            buddy = buddies[0] if buddies else None
            preferences_now = self.persona_buddy_controller.current_preferences()
            region = buddy.region if buddy is not None else None
            controls = {}
            if buddy is not None:
                for control_id in (
                    "persona-buddy-drag-handle",
                    "persona-buddy-collapse",
                    "persona-buddy-close",
                ):
                    control = buddy.query_one(f"#{control_id}")
                    controls[control_id] = {
                        "x": control.region.x,
                        "y": control.region.y,
                        "width": control.region.width,
                        "height": control.region.height,
                    }
            modal_surfaces = list(self.screen.query(ModalHitSurface))
            modal_region = modal_surfaces[0].region if modal_surfaces else None
            modal_target = None
            if modal_region is not None:
                modal_target, _ = self.screen.get_widget_at(
                    modal_region.x + max(0, modal_region.width // 2),
                    modal_region.y + max(0, modal_region.height // 2),
                )
            payload = {
                "capture_released": self.mouse_captured is None,
                "collapsed": preferences_now.collapsed,
                "controls": controls,
                "focus_guard": self.focus_guard_observed,
                "geometry": asdict(preferences_now.geometry),
                "loaded_geometry": self.initial_geometry,
                "modal_hits": self.modal_hits,
                "modal_hit_target": getattr(modal_target, "id", None),
                "modal_region": (
                    {
                        "x": modal_region.x,
                        "y": modal_region.y,
                        "width": modal_region.width,
                        "height": modal_region.height,
                    }
                    if modal_region is not None
                    else None
                ),
                "modal_timer_fired": self.modal_timer_fired,
                "navigation_count": self.navigation_count,
                "open": preferences_now.open,
                "painted": "Buddy"
                in "\n".join(
                    strip.text for strip in screen._compositor.render_strips()
                ),
                "region": (
                    {
                        "x": region.x,
                        "y": region.y,
                        "width": region.width,
                        "height": region.height,
                    }
                    if region is not None
                    else None
                ),
                "view_present": buddy is not None,
                "screen_generation": screen.persona_buddy_view_generation,
                "viewport_clamped": (
                    region is not None
                    and region.x >= 0
                    and region.y >= 0
                    and region.right <= self.size.width
                    and region.bottom <= self.size.height
                ),
            }
            report_path.write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )

    ProbeApp().run(mouse=True)
    return 0


def _set_size(fd: int, columns: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


def _wait_for_output(fd: int, process: subprocess.Popen[bytes], needle: bytes) -> bytes:
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    captured = bytearray()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        try:
            captured.extend(os.read(fd, 65536))
        except OSError:
            break
        if needle in captured:
            return bytes(captured)
    tail = bytes(captured[-4000:]).decode("utf-8", errors="replace")
    raise RuntimeError(f"persona_buddy_terminal_not_ready\n{tail}")


def _drain_for(fd: int, duration: float) -> bytes:
    """Drain terminal paint while allowing a bounded interval between events."""

    deadline = time.monotonic() + duration
    captured = bytearray()
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [fd], [], [], min(0.02, deadline - time.monotonic())
        )
        if ready:
            captured.extend(os.read(fd, 65536))
    return bytes(captured)


def _send_mouse(fd: int, code: int, x: int, y: int, *, release: bool = False) -> None:
    suffix = "m" if release else "M"
    os.write(fd, f"\x1b[<{code};{x + 1};{y + 1}{suffix}".encode())


def _center(region: dict[str, int]) -> tuple[int, int]:
    return (
        region["x"] + max(0, region["width"] // 2),
        region["y"] + max(0, region["height"] // 2),
    )


def _run_child(
    *,
    root: Path,
    preferences: Path,
    report: Path,
    drive: bool,
) -> dict[str, Any]:
    master, slave = pty.openpty()
    _set_size(slave, 80, 24)
    isolated = preferences.parent
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(isolated / "home"),
            "XDG_CONFIG_HOME": str(isolated / "config"),
            "XDG_DATA_HOME": str(isolated / "data"),
            "TLDW_CONFIG_PATH": str(isolated / "config" / "config.toml"),
            "PYTHONPATH": str(root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    for directory in (isolated / "home", isolated / "config", isolated / "data"):
        directory.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(preferences),
            str(report),
        ],
        cwd=root,
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    try:
        _wait_for_output(master, process, b"Persona Buddy")
        initial_deadline = time.monotonic() + 2.0
        while not report.exists() and time.monotonic() < initial_deadline:
            time.sleep(0.02)
        if not report.exists():
            raise RuntimeError("persona_buddy_terminal_initial_report_missing")

        def wait_for_report(predicate: Any, *, timeout: float = 2.0) -> dict[str, Any]:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and process.poll() is None:
                payload = json.loads(report.read_text(encoding="utf-8"))
                if predicate(payload):
                    return payload
                _drain_for(master, 0.02)
            raise RuntimeError("persona_buddy_terminal_report_predicate_timeout")

        initial = json.loads(report.read_text(encoding="utf-8"))
        observed = {
            "drag": False,
            "mouse_resize": False,
            "keyboard": False,
            "fold": False,
            "reopen": False,
            "close": False,
            "navigation_view": False,
            "paint": initial["painted"],
            "viewport_clamp": False,
        }
        if drive:
            drag_x, drag_y = _center(initial["controls"]["persona-buddy-drag-handle"])
            move_x = max(1, drag_x - 10)
            move_y = max(1, drag_y - 5)
            _send_mouse(master, 0, drag_x, drag_y)
            _drain_for(master, 0.10)
            _send_mouse(master, 32, move_x, move_y)
            _drain_for(master, 0.10)
            _send_mouse(master, 0, move_x, move_y, release=True)
            _drain_for(master, 0.25)
            after_drag = json.loads(report.read_text(encoding="utf-8"))
            observed["drag"] = after_drag["geometry"]["x"] < initial["region"]["x"]

            region = after_drag["region"]
            corner_x, corner_y = (
                region["x"] + region["width"] - 1,
                region["y"] + region["height"] - 1,
            )
            _send_mouse(master, 0, corner_x, corner_y)
            _drain_for(master, 0.10)
            _send_mouse(master, 32, corner_x + 5, corner_y + 2)
            _drain_for(master, 0.10)
            _send_mouse(master, 0, corner_x + 5, corner_y + 2, release=True)
            _drain_for(master, 0.30)
            after_resize = json.loads(report.read_text(encoding="utf-8"))
            observed["mouse_resize"] = (
                after_resize["geometry"]["width"] > after_drag["geometry"]["width"]
                and after_resize["geometry"]["height"]
                > after_drag["geometry"]["height"]
            )

            os.write(master, b"hH")
            _drain_for(master, 0.35)
            keyboard = json.loads(report.read_text(encoding="utf-8"))
            keyboard_changed = (
                keyboard["geometry"]["x"] == after_resize["geometry"]["x"] - 1
                and keyboard["geometry"]["width"]
                == after_resize["geometry"]["width"] - 1
            )
            os.write(master, b"0")
            _drain_for(master, 0.35)
            reset = json.loads(report.read_text(encoding="utf-8"))
            observed["keyboard"] = (
                keyboard_changed
                and reset["geometry"]["width"] == 28
                and reset["geometry"]["height"] == 12
            )

            fold_x, fold_y = _center(reset["controls"]["persona-buddy-collapse"])
            _send_mouse(master, 0, fold_x, fold_y)
            _drain_for(master, 0.10)
            _send_mouse(master, 0, fold_x, fold_y, release=True)
            _drain_for(master, 0.35)
            folded = json.loads(report.read_text(encoding="utf-8"))
            observed["fold"] = folded["collapsed"]
            reopen_x, reopen_y = _center(folded["controls"]["persona-buddy-collapse"])
            _send_mouse(master, 0, reopen_x, reopen_y)
            _drain_for(master, 0.10)
            _send_mouse(master, 0, reopen_x, reopen_y, release=True)
            _drain_for(master, 0.35)
            reopened = json.loads(report.read_text(encoding="utf-8"))
            observed["reopen"] = not reopened["collapsed"]

            close_x, close_y = _center(reopened["controls"]["persona-buddy-close"])
            _send_mouse(master, 0, close_x, close_y)
            _drain_for(master, 0.10)
            _send_mouse(master, 0, close_x, close_y, release=True)
            close_output = _drain_for(master, 0.60)
            closed = json.loads(report.read_text(encoding="utf-8"))
            observed["close"] = not closed["open"] and not closed["view_present"]
            if process.poll() is not None:
                raise RuntimeError(close_output.decode("utf-8", errors="replace"))
            os.write(master, b"o")
            _drain_for(master, 0.60)

            os.write(master, b"m")
            modal_report = wait_for_report(
                lambda payload: payload["modal_region"] is not None
            )
            modal_region = modal_report["modal_region"]
            modal_col = modal_region["x"] + max(1, modal_region["width"] // 2)
            modal_row = modal_region["y"] + max(1, modal_region["height"] // 2)
            _send_mouse(master, 0, modal_col, modal_row)
            _drain_for(master, 0.10)
            _send_mouse(master, 0, modal_col, modal_row, release=True)
            _drain_for(master, 0.55)
            navigated = json.loads(report.read_text(encoding="utf-8"))
            observed["navigation_view"] = navigated["view_present"]
            _set_size(master, 60, 18)
            process.send_signal(signal.SIGWINCH)
            _drain_for(master, 0.55)
            resized = json.loads(report.read_text(encoding="utf-8"))
            observed["viewport_clamp"] = resized["viewport_clamped"]
        else:
            _drain_for(master, 0.30)
        deadline = time.monotonic() + _TIMEOUT_SECONDS
        while (
            not report.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            select.select([master], [], [], 0.05)
        if not report.exists():
            captured = bytearray()
            while select.select([master], [], [], 0.05)[0]:
                captured.extend(os.read(master, 65536))
            tail = bytes(captured[-6000:]).decode("utf-8", errors="replace")
            raise RuntimeError(f"persona_buddy_terminal_child_failed\n{tail}")
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["initial_region"] = initial["region"]
        payload["observed"] = observed
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
        return payload
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def _parent(report_output: Path | None = None) -> int:
    if os.name == "nt":
        print("SKIP persona_buddy_terminal windows_no_posix_pty")
        return 0
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="persona-buddy-terminal-") as temporary:
        isolated = Path(temporary)
        preferences = isolated / "persona_buddy.json"
        first = _run_child(
            root=root,
            preferences=preferences,
            report=isolated / "first.json",
            drive=True,
        )
        restored = json.loads(preferences.read_text(encoding="utf-8"))
        restored["open"] = True
        preferences.write_text(json.dumps(restored, sort_keys=True), encoding="utf-8")
        second = _run_child(
            root=root,
            preferences=preferences,
            report=isolated / "second.json",
            drive=False,
        )
        checks = {
            "drag": first["observed"]["drag"],
            "mouse_resize": first["observed"]["mouse_resize"],
            "keyboard": first["observed"]["keyboard"],
            "fold": first["observed"]["fold"],
            "reopen": first["observed"]["reopen"],
            "close": first["observed"]["close"],
            "focus": first["focus_guard"],
            "modal_hit_testing": first["modal_hits"] >= 1,
            "navigation": (
                first["navigation_count"] == 1
                and first["screen_generation"] >= 1
                and first["observed"]["navigation_view"]
            ),
            "viewport_clamp": first["observed"]["viewport_clamp"],
            "capture_release": first["capture_released"],
            "paint": first["observed"]["paint"],
            "geometry_restore": second["loaded_geometry"] == first["geometry"],
        }
        result = {"checks": checks, "first": first, "second": second}
        if report_output is not None:
            report_output.write_text(
                json.dumps(result, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps(result, sort_keys=True))
        if not all(checks.values()):
            print("FAIL persona_buddy_terminal")
            return 1
        print("PASS persona_buddy_terminal")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("preferences", nargs="?", type=Path)
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--report", dest="parent_report", type=Path)
    arguments = parser.parse_args()
    if arguments.child:
        if arguments.preferences is None or arguments.report is None:
            return 2
        return _child(arguments.preferences, arguments.report)
    return _parent(arguments.parent_report)


if __name__ == "__main__":
    raise SystemExit(main())
