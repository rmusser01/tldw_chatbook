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

    from dataclasses import asdict

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
            Binding("q", "probe_finish", "Finish", priority=True),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.persona_buddy_controller = PersonaBuddyController(
                preferences=preferences,
                preference_writer=write_preferences,
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
            self.set_timer(0.8, self.action_probe_modal)

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


def _drain_for(fd: int, duration: float) -> None:
    """Drain terminal paint while allowing a bounded interval between events."""

    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [fd], [], [], min(0.02, deadline - time.monotonic())
        )
        if ready:
            os.read(fd, 65536)


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
        initial = json.loads(report.read_text(encoding="utf-8"))
        if drive:
            # Compute true terminal-cell coordinates from the child's painted
            # region (SGR is 1-based), while still sending the real
            # terminal shape whose Textual MouseDown has widget=None.
            region = initial["region"]
            down_col = region["x"] + 4
            down_row = region["y"] + 3
            move_col = max(2, down_col - 10)
            move_row = max(2, down_row - 5)
            os.write(master, f"\x1b[<0;{down_col};{down_row}M".encode())
            _drain_for(master, 0.10)
            os.write(master, f"\x1b[<32;{move_col};{move_row}M".encode())
            _drain_for(master, 0.10)
            os.write(master, f"\x1b[<0;{move_col};{move_row}m".encode())
            _drain_for(master, 0.25)
            os.write(master, b"hHcc")
            _drain_for(master, 0.90)
            modal_report = json.loads(report.read_text(encoding="utf-8"))
            modal_region = modal_report["modal_region"]
            modal_col = modal_region["x"] + max(1, modal_region["width"] // 2)
            modal_row = modal_region["y"] + max(1, modal_region["height"] // 2)
            os.write(master, f"\x1b[<0;{modal_col};{modal_row}M".encode())
            _drain_for(master, 0.10)
            os.write(master, f"\x1b[<0;{modal_col};{modal_row}m".encode())
            _drain_for(master, 0.55)
            _set_size(master, 60, 18)
            process.send_signal(signal.SIGWINCH)
            _drain_for(master, 0.55)
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
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
        return payload
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def _parent() -> int:
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
        second = _run_child(
            root=root,
            preferences=preferences,
            report=isolated / "second.json",
            drive=False,
        )
        checks = {
            "drag": (
                first["geometry"]["x"] <= first["initial_region"]["x"] - 5
                and first["geometry"]["y"] <= first["initial_region"]["y"] - 3
            ),
            "resize_keys": first["geometry"]["width"] != 28,
            "focus": first["focus_guard"],
            "modal_hit_testing": first["modal_hits"] >= 1,
            "navigation": (
                first["navigation_count"] == 1
                and first["screen_generation"] >= 1
                and first["view_present"]
            ),
            "viewport_clamp": first["viewport_clamped"],
            "capture_release": first["capture_released"],
            "paint": first["painted"],
            "geometry_restore": second["loaded_geometry"] == first["geometry"],
            "open_and_expanded": first["open"] and not first["collapsed"],
        }
        result = {"checks": checks, "first": first, "second": second}
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
    arguments = parser.parse_args()
    if arguments.child:
        if arguments.preferences is None or arguments.report is None:
            return 2
        return _child(arguments.preferences, arguments.report)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
