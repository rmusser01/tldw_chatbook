#!/usr/bin/env python3
"""Bounded POSIX-PTY verification for the native Persona Buddy interactions."""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

if os.name != "nt":
    import fcntl
    import pty
    import termios
else:  # pragma: no cover - exercised by the Windows CLI smoke contract
    fcntl = None
    pty = None
    termios = None

_TIMEOUT_SECONDS = 12.0
_DIAGNOSTIC_BYTES = 16_000
# One 80x24 Textual full repaint is normally tens of KiB. This ceiling keeps the
# exact state-local evidence bounded while leaving generous room for styled cells.
_CAPTURE_BYTES = 256 * 1024
_CAPTURE_FORBIDDEN = (
    b"/private/",
    b"/tmp/",
    b"/var/",
    b"/home/",
    b"/users/",
    b"\\users\\",
    b"tldw_config_path",
    b"config.toml",
    b"provider_inventory",
    b"api_settings",
)
_CAPTURE_NAMES = (
    "normal.ansi",
    "alert.ansi",
    "folded.ansi",
    "constrained.ansi",
)
_CHECK_NAMES = (
    "drag",
    "mouse_resize",
    "keyboard",
    "fold",
    "reopen",
    "close",
    "focus",
    "modal_hit_testing",
    "modal_resume",
    "modal_close_replay",
    "navigation",
    "viewport_clamp",
    "compact_controls",
    "compact_move_restore",
    "capture_release",
    "paint",
    "geometry_restore",
    "graceful_exit",
    "pet_only_normal",
    "fixed_alert_replaces_pet",
    "real_folded_thumbnail",
    "constrained_two_icons",
)


def _validated_cli_path(value: str) -> Path:
    """Confine explicit probe paths to the workspace or platform temp roots."""

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tldw_chatbook.Utils.path_validation import validate_path

    roots = (root, Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())
    candidate = Path(value)
    for index, allowed_root in enumerate(roots):
        if index and not candidate.is_absolute():
            continue
        try:
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (allowed_root / candidate).resolve()
            )
            resolved.relative_to(allowed_root)
        except ValueError:
            continue
        return validate_path(resolved, allowed_root, redact_paths=True)
    raise argparse.ArgumentTypeError("persona_buddy_probe_path_invalid")


class _ProbeChildFailure(RuntimeError):
    """Structured child-process failure retained until evidence is durable."""

    def __init__(
        self,
        *,
        category: str,
        phase: str,
        child_return_code: int,
        diagnostic_tail: str,
        checks: dict[str, bool] | None = None,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.phase = phase
        self.child_return_code = child_return_code
        self.diagnostic_tail = diagnostic_tail
        self.checks = checks


def _atomic_write_text(
    path: Path, value: str, *, inject_replace_failure: bool = False
) -> None:
    """Replace one evidence file only after its complete contents are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if inject_replace_failure:
            raise OSError("persona_buddy_terminal_report_publish_injected")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    """Atomically replace one exact terminal byte-stream artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    inject_replace_failure: bool = False,
) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, sort_keys=True),
        inject_replace_failure=inject_replace_failure,
    )


def _bounded_utf8_tail(value: str, limit: int = _DIAGNOSTIC_BYTES) -> str:
    """Retain the diagnostic category and a UTF-8-safe tail within ``limit`` bytes."""

    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return encoded.decode("utf-8")
    category, separator, _ = value.partition("\n")
    prefix = (category + separator).encode("utf-8", errors="replace")
    if not separator or len(prefix) >= limit:
        return encoded[:limit].decode("utf-8", errors="ignore")
    tail = encoded[-(limit - len(prefix)) :].decode("utf-8", errors="ignore")
    return prefix.decode("utf-8") + tail


def _publish_capture_group(
    staged_captures: Path,
    capture_output: Path,
    *,
    inject_failure: bool = False,
) -> None:
    """Publish the managed capture set, rolling back this operation on failure."""

    published: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".persona-buddy-publish-", dir=capture_output
        ) as temporary:
            publication = Path(temporary)
            for name in _CAPTURE_NAMES:
                _atomic_write_bytes(
                    publication / name,
                    (staged_captures / name).read_bytes(),
                )
            for index, name in enumerate(_CAPTURE_NAMES):
                if inject_failure and index == 2:
                    raise OSError("persona_buddy_terminal_capture_publish_injected")
                target = capture_output / name
                (publication / name).replace(target)
                published.append(target)
    except Exception as error:
        for target in published:
            target.unlink(missing_ok=True)
        raise _ProbeChildFailure(
            category="persona_buddy_terminal_capture_publish",
            phase="parent:capture",
            child_return_code=0,
            diagnostic_tail=(
                f"{type(error).__name__}: persona_buddy_terminal_capture_publish"
            ),
        ) from error


def _child(preferences_path: Path, report_path: Path) -> int:
    """Run the production-CSS child application inside the allocated PTY."""

    from dataclasses import asdict, replace
    from types import SimpleNamespace

    from rich.text import Text
    from textual import events
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.errors import NoWidget
    from textual.screen import ModalScreen
    from textual.widgets import Input, Static

    from tldw_chatbook.Persona_Buddy import (
        PersonaBuddyController,
        PersonaBuddyPreferences,
        PersonaBuddySelection,
        PersonaBuddyVisualSnapshot,
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
            Binding("r", "resume_probe_buddy", "Resume Buddy", priority=True),
            Binding("x", "close_probe_buddy", "Close Buddy", priority=True),
        ]

        def compose(self) -> ComposeResult:
            yield ModalHitSurface("MODAL BLOCKER", id="terminal-modal-blocker")

        def on_mount(self) -> None:
            self.call_after_refresh(self._publish_ready)

        def on_resize(self, _event: events.Resize) -> None:
            self.call_after_refresh(self._publish_ready)

        def _publish_ready(self) -> None:
            surfaces = list(self.query(ModalHitSurface))
            if not surfaces:
                return
            region = surfaces[0].region
            if region.width <= 0 or region.height <= 0:
                return
            self.app.modal_ready = True
            self.app._write_probe_report()

        def action_close_probe_modal(self) -> None:
            self.dismiss()

        def action_resume_probe_buddy(self) -> None:
            self.app.persona_buddy_controller.invalidate_profile()
            self.dismiss()

        def action_close_probe_buddy(self) -> None:
            controller = self.app.persona_buddy_controller
            revision = controller.apply_preferences_patch(open=False)
            self.app.run_worker(
                controller.persist_preferences_revision(revision),
                group="persona-buddy-preferences",
            )
            self.dismiss()

        def on_unmount(self) -> None:
            self.app.modal_ready = False

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
            Binding("p", "probe_focus_buddy", "Focus Buddy", priority=True),
            Binding("q", "probe_finish", "Finish", priority=True),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.persona_buddy_controller = PersonaBuddyController(
                preferences=preferences,
                preference_writer=write_preferences,
            )

            self.resolution_calls = 0

            async def resolve_probe_visual(*, cols: int, lines: int):
                self.resolution_calls += 1
                snapshot = self.persona_buddy_controller.snapshot()
                collapsed = bool(snapshot.collapsed)
                art = (
                    ("   /\\_/\\  ", "  ( o.o ) ", "   > ^ <  ", "    \\_/   ")
                    if collapsed
                    else ("  /\\_/\\   ", " ( o.o )  ", "  > ^ <   ", " .-~~~-.  ")
                )
                mode = "folded" if collapsed else "normal"
                frame = SimpleNamespace(
                    cache_identity=f"terminal-probe-{mode}",
                    graph_identity=None,
                    asset_id=1,
                    asset_key=f"terminal-probe-{mode}",
                    asset_sha256=f"terminal-probe-{mode}",
                    manifest_frame_index=0,
                    selected_frame=0,
                    duration_ms=1000,
                    width=10,
                    height=8,
                    paint_digest=f"terminal-probe-{mode}",
                    renderable=Text("\n".join(art)),
                )
                return PersonaBuddyVisualSnapshot(
                    available=True,
                    reason=None,
                    source="local",
                    persona_id="terminal-probe",
                    persona_revision=1,
                    requested_state=snapshot.state,
                    resolved_state=snapshot.state,
                    animation_id=f"terminal-probe-{mode}",
                    graph_identity=None,
                    cache_identity=None,
                    frames=(frame,),
                    frame_rate=None,
                    loop=False,
                    animate=False,
                )

            self.persona_buddy_controller.resolve_current_visual = resolve_probe_visual
            self.modal_hits = 0
            self.navigation_count = 0
            self.initial_geometry = loaded_geometry
            self.focus_guard_observed = False
            self.modal_timer_fired = False
            self.modal_ready = False
            self.graceful_exit_requested = False
            self.alert_token = None

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
                self._request_full_repaint,
            )
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGUSR2,
                self.action_probe_finish,
            )
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGALRM,
                self.action_probe_alert,
            )
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGHUP,
                self.action_probe_idle,
            )
            asyncio.get_running_loop().add_signal_handler(
                signal.SIGURG,
                self.action_probe_focus_buddy,
            )
            self.call_after_refresh(self._capture_focus_guard)
            self.set_interval(0.10, self._write_probe_report)

        def _request_full_repaint(self) -> None:
            """Force one whole-screen repaint for exact external evidence."""

            self.screen.refresh(layout=True, repaint=True)
            self.call_after_refresh(self._write_probe_report)

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
            self.modal_ready = False
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

        def action_probe_alert(self) -> None:
            if self.alert_token is None:
                self.alert_token = self.persona_buddy_controller.acquire_state(
                    source="console",
                    owner="terminal-probe",
                    state="approval_needed",
                )
            buddies = list(self.screen.query(PersonaBuddyWidget))
            if buddies:
                buddies[0].refresh_from_controller()

        def action_probe_idle(self) -> None:
            token = self.alert_token
            if token is not None:
                self.persona_buddy_controller.release_state(token=token)
                self.alert_token = None
            buddies = list(self.screen.query(PersonaBuddyWidget))
            if buddies:
                buddies[0].refresh_from_controller()

        def action_probe_focus_buddy(self) -> None:
            buddies = list(self.screen.query(PersonaBuddyWidget))
            if buddies:
                buddies[0].focus(scroll_visible=False)

        def action_probe_finish(self) -> None:
            self.graceful_exit_requested = True
            self._write_probe_report()
            self.exit()

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
            frame = (
                buddy.query_one("#persona-buddy-frame", Static)
                if buddy is not None
                else None
            )
            frame_region = frame.region if frame is not None else None
            controls = {}
            if buddy is not None:
                for control_id in (
                    "persona-buddy-collapse",
                    "persona-buddy-close",
                ):
                    control = buddy.query_one(f"#{control_id}")
                    controls[control_id] = {
                        "display": control.display,
                        "x": control.region.x,
                        "y": control.region.y,
                        "width": control.region.width,
                        "height": control.region.height,
                        "label": str(getattr(control, "label", "")),
                    }
            strips = screen._compositor.render_strips()
            rows = tuple(strip.text for strip in strips)
            painted_text = "\n".join(rows)
            frame_painted_cells = 0
            if frame_region is not None:
                control_regions = tuple(
                    control.region for control in buddy.query(".persona-buddy-control")
                )
                for y in range(frame_region.y, frame_region.bottom):
                    for x in range(frame_region.x, frame_region.right):
                        if any(control.contains(x, y) for control in control_regions):
                            continue
                        if y < len(rows) and x < len(rows[y]) and rows[y][x].strip():
                            frame_painted_cells += 1
            collapse_control = controls.get("persona-buddy-collapse", {})
            close_control = controls.get("persona-buddy-close", {})
            icon_controls = bool(
                collapse_control.get("display")
                and close_control.get("display")
                and collapse_control.get("label")
                in ({"▴"} if preferences_now.collapsed else {"▾"})
                and close_control.get("label") == "×"
            )
            compact = bool(
                buddy is not None and buddy.has_class("persona-buddy-compact")
            )
            collapsed = bool(preferences_now.collapsed)
            actionable_alert = self.persona_buddy_controller.snapshot().state
            accepted = getattr(buddy, "_accepted_render", None)
            default_words_absent = not any(
                word in painted_text
                for word in (
                    "Persona Buddy",
                    "Drag",
                    "Fold",
                    "Close",
                    "State",
                    "Visual pending",
                    "hjkl move",
                    "HJKL size",
                )
            )
            pet_only_normal = bool(
                buddy is not None
                and actionable_alert == "idle"
                and not collapsed
                and not compact
                and accepted is not None
                and not accepted.collapsed
                and frame_painted_cells > 0
                and icon_controls
                and default_words_absent
            )
            fixed_alert_replaces_pet = bool(
                buddy is not None
                and actionable_alert == "approval_needed"
                and frame is not None
                and frame.has_class("persona-buddy-alert")
                and str(frame.renderable) == "Approval needed"
                and "Approval" in painted_text
                and "needed" in painted_text
                and "o.o" not in painted_text
                and icon_controls
            )
            real_folded_thumbnail = bool(
                buddy is not None
                and collapsed
                and not compact
                and accepted is not None
                and accepted.collapsed
                and frame_painted_cells > 0
                and "o.o" in painted_text
                and icon_controls
            )
            constrained_two_icons = bool(
                buddy is not None
                and compact
                and frame is not None
                and not frame.display
                and icon_controls
                and sum(bool(control["display"]) for control in controls.values()) == 2
            )
            modal_surfaces = list(self.screen.query(ModalHitSurface))
            modal_region = modal_surfaces[0].region if modal_surfaces else None
            if modal_region is not None and (
                modal_region.width <= 0 or modal_region.height <= 0
            ):
                modal_region = None
            modal_target = None
            if modal_region is not None:
                try:
                    modal_target, _ = self.screen.get_widget_at(
                        modal_region.x + max(0, modal_region.width // 2),
                        modal_region.y + max(0, modal_region.height // 2),
                    )
                except NoWidget:
                    modal_region = None
            payload = {
                "capture_released": self.mouse_captured is None,
                "interaction_active": bool(
                    buddy is not None and getattr(buddy, "_interaction", None)
                ),
                "resize_active": bool(
                    buddy is not None
                    and getattr(buddy, "_interaction", None)
                    and buddy._interaction[0] == "resize"
                ),
                "buddy_focused": buddy is not None and screen.focused is buddy,
                "collapsed": collapsed,
                "controls": controls,
                "focus_guard": self.focus_guard_observed,
                "geometry": asdict(preferences_now.geometry),
                "working_geometry": (
                    asdict(buddy._working_preferences.geometry)
                    if buddy is not None
                    else None
                ),
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
                "modal_ready": self.modal_ready,
                "navigation_count": self.navigation_count,
                "open": preferences_now.open,
                "resolution_calls": self.resolution_calls,
                "compact": compact,
                "pet_only_normal": pet_only_normal,
                "fixed_alert_replaces_pet": fixed_alert_replaces_pet,
                "real_folded_thumbnail": real_folded_thumbnail,
                "constrained_two_icons": constrained_two_icons,
                "frame_painted_cells": frame_painted_cells,
                "frame_region": (
                    {
                        "x": frame_region.x,
                        "y": frame_region.y,
                        "width": frame_region.width,
                        "height": frame_region.height,
                    }
                    if frame_region is not None
                    else None
                ),
                "graceful_exit_requested": self.graceful_exit_requested,
                "painted": frame_painted_cells > 0,
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
                "viewport": {"width": self.size.width, "height": self.size.height},
                "screen_generation": screen.persona_buddy_view_generation,
                "viewport_clamped": (
                    region is not None
                    and region.x >= 0
                    and region.y >= 0
                    and region.right <= self.size.width
                    and region.bottom <= self.size.height
                ),
            }
            _atomic_write_json(report_path, payload)

    ProbeApp().run(mouse=True)
    return 0


def _set_size(fd: int, columns: int, rows: int) -> None:
    if fcntl is None or termios is None:  # pragma: no cover - POSIX-only caller
        raise RuntimeError("persona_buddy_terminal_posix_pty_unavailable")
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


def _read_pty(fd: int) -> bytes:
    """Read one PTY chunk, treating Linux's EIO-on-slave-close as clean EOF."""

    try:
        return os.read(fd, 65536)
    except OSError as error:
        if error.errno == errno.EIO:
            return b""
        raise


def _drain_for(fd: int, duration: float) -> bytes:
    """Drain terminal paint while allowing a bounded interval between events."""

    deadline = time.monotonic() + duration
    captured = bytearray()
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [fd], [], [], min(0.02, deadline - time.monotonic())
        )
        if ready:
            chunk = _read_pty(fd)
            if not chunk:
                break
            captured.extend(chunk)
    return bytes(captured)


def _drain_until_quiet(
    fd: int,
    *,
    timeout: float = 0.50,
    quiet_seconds: float = 0.05,
    max_bytes: int | None = None,
) -> bytes:
    """Drain until the PTY has stayed quiet, bounded by one hard deadline."""

    deadline = time.monotonic() + timeout
    captured = bytearray()
    while time.monotonic() < deadline:
        wait = min(quiet_seconds, deadline - time.monotonic())
        ready, _, _ = select.select([fd], [], [], wait)
        if not ready:
            break
        chunk = _read_pty(fd)
        if not chunk:
            break
        if max_bytes is not None and len(captured) + len(chunk) > max_bytes:
            raise ValueError("persona_buddy_terminal_capture_too_large")
        captured.extend(chunk)
    return bytes(captured)


def _ansi_sequences_are_complete(value: bytes) -> bool:
    """Return whether every ESC sequence ends at a valid control boundary."""

    index = 0
    length = len(value)
    while index < length:
        if value[index] != 0x1B:
            index += 1
            continue
        index += 1
        if index >= length:
            return False
        introducer = value[index]
        index += 1
        if introducer == ord("["):
            while index < length and not 0x40 <= value[index] <= 0x7E:
                index += 1
            if index >= length:
                return False
            index += 1
        elif introducer in (ord("]"), ord("P"), ord("_"), ord("^")):
            while index < length:
                if value[index] == 0x07:
                    index += 1
                    break
                if (
                    value[index] == 0x1B
                    and index + 1 < length
                    and value[index + 1] == ord("\\")
                ):
                    index += 2
                    break
                index += 1
            else:
                return False
        else:
            while 0x20 <= introducer <= 0x2F:
                if index >= length:
                    return False
                introducer = value[index]
                index += 1
            if not 0x30 <= introducer <= 0x7E:
                return False
    return True


def _validate_terminal_capture(value: bytes) -> None:
    """Fail closed unless one exact terminal state is bounded and replay-safe."""

    if not value:
        raise ValueError("persona_buddy_terminal_capture_empty")
    if len(value) > _CAPTURE_BYTES:
        raise ValueError("persona_buddy_terminal_capture_too_large")
    value.decode("utf-8")
    lowered = value.lower()
    if any(marker in lowered for marker in _CAPTURE_FORBIDDEN):
        raise ValueError("persona_buddy_terminal_capture_private_content")
    if not _ansi_sequences_are_complete(value):
        raise ValueError("persona_buddy_terminal_capture_incomplete_ansi")


def _capture_fresh_repaint(fd: int, request_repaint: Any | None = None) -> bytes:
    """Discard old PTY traffic, request a full repaint, and retain only that state."""

    _drain_until_quiet(fd)
    if request_repaint is None:
        os.write(fd, b"\x0c")
    else:
        request_repaint()
    captured = _drain_until_quiet(fd, max_bytes=_CAPTURE_BYTES)
    _validate_terminal_capture(captured)
    return captured


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
    phase: str,
    capture_dir: Path | None = None,
    inject_child_failure: bool = False,
    inject_startup_noise: bool = False,
) -> dict[str, Any]:
    if pty is None:  # pragma: no cover - parent skips this path on Windows
        raise RuntimeError("persona_buddy_terminal_posix_pty_unavailable")
    master, slave = pty.openpty()
    _set_size(slave, 80, 24)
    isolated = preferences.parent
    environment = os.environ.copy()
    environment.pop("NO_COLOR", None)
    environment.update(
        {
            "HOME": str(isolated / "home"),
            "XDG_CONFIG_HOME": str(isolated / "config"),
            "XDG_DATA_HOME": str(isolated / "data"),
            "TLDW_CONFIG_PATH": str(isolated / "config" / "config.toml"),
            "PYTHONPATH": str(root),
            "PYTHONUNBUFFERED": "1",
            "TERM": "xterm-256color",
        }
    )
    for directory in (isolated / "home", isolated / "config", isolated / "data"):
        directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve())]
    if inject_startup_noise:
        command.append("--inject-startup-noise")
    if inject_child_failure:
        command.append("--inject-child-failure")
    command.extend(("--child", str(preferences), str(report)))
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    current_phase = f"{phase}:startup"
    terminal_output = bytearray()

    def drain(duration: float = 0.02) -> bytes:
        chunk = _drain_for(master, duration)
        terminal_output.extend(chunk)
        return chunk

    def capture(name: str) -> None:
        if capture_dir is None:
            return
        _atomic_write_bytes(
            capture_dir / name,
            _capture_fresh_repaint(master, lambda: process.send_signal(signal.SIGUSR1)),
        )

    try:
        initial_deadline = time.monotonic() + _TIMEOUT_SECONDS
        startup_output = bytearray()
        initial = None
        while time.monotonic() < initial_deadline and process.poll() is None:
            if report.exists():
                candidate = json.loads(report.read_text(encoding="utf-8"))
                region = candidate.get("region")
                if (
                    candidate.get("pet_only_normal")
                    and region is not None
                    and region["width"] > 0
                    and region["height"] > 0
                ):
                    initial = candidate
                    break
            chunk = drain()
            startup_output.extend(chunk)
        if initial is None:
            tail = bytes(startup_output[-4000:]).decode("utf-8", errors="replace")
            raise RuntimeError(f"persona_buddy_terminal_initial_report_missing\n{tail}")

        def wait_for_report(predicate: Any, *, timeout: float = 2.0) -> dict[str, Any]:
            deadline = time.monotonic() + timeout
            captured = bytearray()
            payload = json.loads(report.read_text(encoding="utf-8"))
            while time.monotonic() < deadline and process.poll() is None:
                payload = json.loads(report.read_text(encoding="utf-8"))
                if predicate(payload):
                    return payload
                chunk = drain()
                captured.extend(chunk)
            tail = bytes(captured[-2000:]).decode("utf-8", errors="replace")
            raise RuntimeError(
                "persona_buddy_terminal_report_predicate_timeout "
                f"payload={payload!r} terminal_tail={tail!r}"
            )

        current_phase = f"{phase}:interaction"
        observed = {
            "drag": False,
            "mouse_resize": False,
            "keyboard": False,
            "fold": False,
            "reopen": False,
            "close": False,
            "modal_resume": False,
            "modal_close_replay": False,
            "navigation_view": False,
            "paint": initial["painted"],
            "viewport_clamp": False,
            "compact_controls": False,
            "compact_move_restore": False,
            "graceful_exit": False,
            "pet_only_normal": False,
            "fixed_alert_replaces_pet": False,
            "real_folded_thumbnail": False,
            "constrained_two_icons": False,
        }
        observed_regions: dict[str, dict[str, int]] = {}
        if drive:
            observed["pet_only_normal"] = initial["pet_only_normal"]
            observed_regions["normal"] = initial["region"]
            capture("normal.ansi")

            process.send_signal(signal.SIGALRM)
            alert = wait_for_report(lambda payload: payload["fixed_alert_replaces_pet"])
            observed["fixed_alert_replaces_pet"] = True
            observed_regions["alert"] = alert["region"]
            capture("alert.ansi")

            process.send_signal(signal.SIGHUP)
            idle = wait_for_report(lambda payload: payload["pet_only_normal"])

            drag_x, drag_y = _center(idle["frame_region"])
            move_x = max(1, drag_x - 10)
            move_y = max(1, drag_y - 5)
            _send_mouse(master, 0, drag_x, drag_y)
            wait_for_report(
                lambda payload: (
                    payload["interaction_active"] and not payload["resize_active"]
                )
            )
            _send_mouse(master, 32, move_x, move_y)
            wait_for_report(
                lambda payload: payload["region"]["x"] < idle["region"]["x"]
            )
            _send_mouse(master, 0, move_x, move_y, release=True)
            after_drag = wait_for_report(
                lambda payload: (
                    payload["capture_released"]
                    and payload["geometry"]["x"] < idle["region"]["x"]
                )
            )
            observed["drag"] = True

            region = after_drag["region"]
            corner_x, corner_y = (
                region["x"] + region["width"] - 1,
                region["y"] + region["height"] - 1,
            )
            _send_mouse(master, 0, corner_x, corner_y)
            wait_for_report(lambda payload: payload["resize_active"])
            _send_mouse(master, 32, corner_x + 5, corner_y + 2)
            wait_for_report(
                lambda payload: (
                    payload["working_geometry"]["width"]
                    > after_drag["geometry"]["width"]
                    and payload["working_geometry"]["height"]
                    > after_drag["geometry"]["height"]
                )
            )
            _send_mouse(master, 0, corner_x + 5, corner_y + 2, release=True)
            after_resize = wait_for_report(
                lambda payload: (
                    payload["capture_released"]
                    and payload["geometry"]["width"] > after_drag["geometry"]["width"]
                    and payload["geometry"]["height"] > after_drag["geometry"]["height"]
                )
            )
            observed["mouse_resize"] = True

            os.write(master, b"hH")
            keyboard = wait_for_report(
                lambda payload: (
                    payload["geometry"]["x"] == after_resize["geometry"]["x"] - 1
                    and payload["geometry"]["width"]
                    == after_resize["geometry"]["width"] - 1
                )
            )
            keyboard_changed = (
                keyboard["geometry"]["x"] == after_resize["geometry"]["x"] - 1
                and keyboard["geometry"]["width"]
                == after_resize["geometry"]["width"] - 1
            )
            os.write(master, b"0")
            reset = wait_for_report(
                lambda payload: (
                    payload["geometry"]["width"] == 28
                    and payload["geometry"]["height"] == 12
                )
            )
            observed["keyboard"] = (
                keyboard_changed
                and reset["geometry"]["width"] == 28
                and reset["geometry"]["height"] == 12
            )

            fold_x, fold_y = _center(reset["controls"]["persona-buddy-collapse"])
            _send_mouse(master, 0, fold_x, fold_y)
            wait_for_report(
                lambda payload: (
                    payload["controls"]["persona-buddy-collapse"]["label"] == "Fold"
                )
            )
            _send_mouse(master, 0, fold_x, fold_y, release=True)
            wait_for_report(
                lambda payload: (
                    payload["collapsed"]
                    and payload["controls"]["persona-buddy-collapse"]["label"] == "Open"
                )
            )
            process.send_signal(signal.SIGURG)
            folded = wait_for_report(lambda payload: payload["real_folded_thumbnail"])
            observed["fold"] = True
            observed["real_folded_thumbnail"] = True
            observed_regions["folded"] = folded["region"]
            capture("folded.ansi")
            reopen_x, reopen_y = _center(folded["controls"]["persona-buddy-collapse"])
            _send_mouse(master, 0, reopen_x, reopen_y)
            wait_for_report(
                lambda payload: (
                    payload["controls"]["persona-buddy-collapse"]["label"] == "Open"
                )
            )
            _send_mouse(master, 0, reopen_x, reopen_y, release=True)
            wait_for_report(
                lambda payload: (
                    not payload["collapsed"]
                    and payload["controls"]["persona-buddy-collapse"]["label"] == "Fold"
                )
            )
            process.send_signal(signal.SIGURG)
            reopened = wait_for_report(lambda payload: payload["pet_only_normal"])
            observed["reopen"] = True

            close_x, close_y = _center(reopened["controls"]["persona-buddy-close"])
            _send_mouse(master, 0, close_x, close_y)
            wait_for_report(
                lambda payload: (
                    payload["controls"]["persona-buddy-close"]["label"] == "Close"
                )
            )
            _send_mouse(master, 0, close_x, close_y, release=True)
            wait_for_report(
                lambda payload: not payload["open"] and not payload["view_present"]
            )
            observed["close"] = True
            os.write(master, b"o")
            reopened_view = wait_for_report(
                lambda payload: payload["open"] and payload["view_present"]
            )

            resume_calls = reopened_view["resolution_calls"]
            os.write(master, b"m")
            wait_for_report(
                lambda payload: (
                    payload["modal_ready"] and payload["modal_region"] is not None
                )
            )
            os.write(master, b"r")
            resumed = wait_for_report(
                lambda payload: (
                    not payload["modal_ready"]
                    and payload["view_present"]
                    and payload["resolution_calls"] > resume_calls
                )
            )
            observed["modal_resume"] = resumed["open"]

            os.write(master, b"m")
            wait_for_report(
                lambda payload: (
                    payload["modal_ready"] and payload["modal_region"] is not None
                )
            )
            os.write(master, b"x")
            modal_closed = wait_for_report(
                lambda payload: (
                    not payload["modal_ready"]
                    and not payload["open"]
                    and not payload["view_present"]
                )
            )
            observed["modal_close_replay"] = not modal_closed["view_present"]

            os.write(master, b"o")
            wait_for_report(lambda payload: payload["open"] and payload["view_present"])

            os.write(master, b"m")
            modal_report = wait_for_report(
                lambda payload: (
                    payload["modal_ready"] and payload["modal_region"] is not None
                )
            )
            modal_region = modal_report["modal_region"]
            modal_col = modal_region["x"] + max(1, modal_region["width"] // 2)
            modal_row = modal_region["y"] + max(1, modal_region["height"] // 2)
            navigated = modal_report
            for attempt in range(2):
                _send_mouse(master, 0, modal_col, modal_row)
                _send_mouse(master, 0, modal_col, modal_row, release=True)
                try:
                    navigated = wait_for_report(
                        lambda payload: (
                            payload["modal_hits"] >= 1
                            and payload["navigation_count"] == 1
                            and payload["view_present"]
                        ),
                        timeout=0.8,
                    )
                    break
                except RuntimeError:
                    if attempt == 1:
                        raise
            observed["navigation_view"] = navigated["view_present"]
            os.write(master, b"p")
            wait_for_report(lambda payload: payload["buddy_focused"])
            _set_size(master, 12, 6)
            process.send_signal(signal.SIGWINCH)
            threshold = wait_for_report(
                lambda payload: (
                    not payload["compact"]
                    and payload["viewport"] == {"width": 12, "height": 6}
                    and payload["viewport_clamped"]
                    and payload["controls"]["persona-buddy-collapse"]["display"]
                    and payload["controls"]["persona-buddy-close"]["display"]
                    and payload["controls"]["persona-buddy-collapse"]["label"] == "▾"
                    and payload["controls"]["persona-buddy-close"]["label"] == "×"
                )
            )
            _set_size(master, 10, 2)
            process.send_signal(signal.SIGWINCH)
            compact_minimal = wait_for_report(
                lambda payload: (
                    payload["viewport"] == {"width": 10, "height": 2}
                    and payload["constrained_two_icons"]
                )
            )
            observed["viewport_clamp"] = compact_minimal["viewport_clamped"]
            observed["compact_controls"] = bool(
                not threshold["compact"] and compact_minimal["compact"]
            )
            observed["constrained_two_icons"] = True
            observed_regions["constrained"] = compact_minimal["region"]
            capture("constrained.ansi")
            compact_preferred = compact_minimal["geometry"]
            compact_display_y = compact_minimal["region"]["y"]
            os.write(master, b"k")
            compact_moved = wait_for_report(
                lambda payload: (
                    payload["geometry"]["y"] == max(0, compact_display_y - 1)
                    and payload["geometry"]["width"] == compact_preferred["width"]
                    and payload["geometry"]["height"] == compact_preferred["height"]
                )
            )
            _set_size(master, 28, 12)
            process.send_signal(signal.SIGWINCH)
            compact_restored = wait_for_report(
                lambda payload: (
                    not payload["compact"]
                    and payload["viewport"] == {"width": 28, "height": 12}
                    and payload["region"]["width"] == initial["region"]["width"]
                    and payload["region"]["height"] == initial["region"]["height"]
                )
            )
            observed["compact_move_restore"] = bool(
                compact_moved["geometry"]["width"] == 28
                and compact_moved["geometry"]["height"] == 12
                and compact_restored["region"]["width"] == initial["region"]["width"]
                and compact_restored["region"]["height"] == initial["region"]["height"]
            )
        else:
            wait_for_report(lambda payload: payload["pet_only_normal"])
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
                chunk = _read_pty(master)
                if not chunk:
                    break
                captured.extend(chunk)
            tail = bytes(captured[-6000:]).decode("utf-8", errors="replace")
            raise RuntimeError(f"persona_buddy_terminal_child_failed\n{tail}")
        process.send_signal(signal.SIGUSR2)
        drain(0.20)
        process.wait(timeout=2)
        payload = json.loads(report.read_text(encoding="utf-8"))
        observed["graceful_exit"] = bool(
            process.returncode == 0 and payload["graceful_exit_requested"]
        )
        payload["initial_region"] = initial["region"]
        payload["observed"] = observed
        payload["observed_regions"] = observed_regions
        return payload
    except Exception as error:
        failure_error = error
        captured = bytearray()
        drain_deadline = time.monotonic() + 0.25
        while time.monotonic() < drain_deadline:
            ready, _, _ = select.select([master], [], [], 0.02)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = _read_pty(master)
            except OSError as read_error:
                failure_error = read_error
                break
            if not chunk:
                break
            captured.extend(chunk)
        child_return_code = process.poll()
        if child_return_code is None:
            process.terminate()
            process.wait(timeout=2)
            child_return_code = process.returncode
        diagnostic = f"{type(failure_error).__name__}: {failure_error}"
        if captured:
            diagnostic += "\n" + bytes(captured).decode("utf-8", errors="replace")
        category = (
            "persona_buddy_terminal_child_exit"
            if child_return_code != 0
            else "persona_buddy_terminal_probe_failure"
        )
        raise _ProbeChildFailure(
            category=category,
            phase=current_phase,
            child_return_code=int(child_return_code),
            diagnostic_tail=_bounded_utf8_tail(diagnostic),
        ) from error
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def _rollback_managed_captures(capture_output: Path | None) -> None:
    """Remove only regular managed artifacts, preserving unrelated directories."""

    if capture_output is None or not capture_output.is_dir():
        return
    for name in _CAPTURE_NAMES:
        target = capture_output / name
        if not target.is_dir():
            target.unlink(missing_ok=True)


def _preflight_atomic_directory(directory: Path, target_name: str) -> None:
    """Prove an atomic target's parent is writable without touching the target."""

    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise OSError("persona_buddy_terminal_output_parent_not_directory")
    descriptor, probe_name = tempfile.mkstemp(
        prefix=f".{target_name}.admission.", suffix=".tmp", dir=directory
    )
    probe = Path(probe_name)
    try:
        os.close(descriptor)
    finally:
        probe.unlink(missing_ok=True)


def _admit_outputs(report_output: Path | None, capture_output: Path | None) -> None:
    """Validate all caller-owned output locations before starting a child."""

    if capture_output is None:
        raise OSError("persona_buddy_terminal_capture_directory_required")
    if capture_output.exists() and not capture_output.is_dir():
        raise OSError("persona_buddy_terminal_capture_directory_not_directory")
    capture_output.mkdir(parents=True, exist_ok=True)
    _preflight_atomic_directory(capture_output, "capture")
    if report_output is not None:
        if report_output.exists() and report_output.is_dir():
            raise OSError("persona_buddy_terminal_report_is_directory")
        _preflight_atomic_directory(report_output.parent, report_output.name)


def _failure_sibling(report_output: Path | None) -> Path:
    if report_output is not None:
        return report_output.with_name(
            f".{report_output.name}.{os.getpid()}.{time.monotonic_ns()}.failure.json"
        )
    return Path(tempfile.gettempdir()) / (
        f"persona-buddy-terminal-{os.getpid()}.{time.monotonic_ns()}.failure.json"
    )


def _persist_structured_failure(
    failure: _ProbeChildFailure,
    *,
    report_output: Path | None,
    root: Path,
    temporary: str | None,
) -> tuple[dict[str, Any], Path]:
    """Persist bounded failure evidence, falling back from an unusable target."""

    diagnostic = failure.diagnostic_tail.replace(str(root), "<REPO_ROOT>")
    if temporary is not None:
        diagnostic = diagnostic.replace(temporary, "<TEMP_ROOT>")
    diagnostic = _bounded_utf8_tail(diagnostic)
    preferred_report = report_output
    if preferred_report is None or preferred_report.is_dir():
        preferred_report = _failure_sibling(report_output)
    diagnostic_path = preferred_report.with_name(
        f"{preferred_report.stem}.diagnostic.log"
    ).resolve()
    try:
        _atomic_write_text(diagnostic_path, diagnostic)
    except OSError:
        diagnostic_path = (
            Path(tempfile.gettempdir())
            / f"persona-buddy-terminal-{os.getpid()}.diagnostic.log"
        ).resolve()
        _atomic_write_text(diagnostic_path, diagnostic)
    supplied_checks = failure.checks or {}
    checks = {name: bool(supplied_checks.get(name, False)) for name in _CHECK_NAMES}
    result = {
        "status": "FAIL",
        "category": failure.category,
        "phase": failure.phase,
        "parent_return_code": 1,
        "child_return_code": failure.child_return_code,
        "diagnostic_tail": diagnostic,
        "diagnostic_artifact": str(diagnostic_path),
        "checks": checks,
        "check_statuses": {
            name: (
                "passed"
                if checks[name]
                else "failed"
                if name in supplied_checks
                else "not_run"
            )
            for name in _CHECK_NAMES
        },
    }
    try:
        _atomic_write_json(preferred_report, result)
    except OSError:
        fallback_report = _failure_sibling(report_output)
        try:
            _atomic_write_json(fallback_report, result)
        except OSError:
            fallback_report = (
                Path(tempfile.gettempdir())
                / f"persona-buddy-terminal-{os.getpid()}.failure.json"
            )
            _atomic_write_json(fallback_report, result)
    return result, diagnostic_path


def _parent(
    report_output: Path | None = None,
    *,
    capture_output: Path | None = None,
    inject_child_failure: bool = False,
    inject_publication_failure: bool = False,
    inject_report_publication_failure: bool = False,
    inject_startup_noise: bool = False,
    inject_check_failure: bool = False,
) -> int:
    if os.name == "nt":
        print("SKIP persona_buddy_terminal windows_no_posix_pty")
        return 0
    root = Path(__file__).resolve().parents[2]
    temporary: str | None = None
    phase = "parent:admission"
    child_return_code = 0
    try:
        _admit_outputs(report_output, capture_output)
        _rollback_managed_captures(capture_output)
        with tempfile.TemporaryDirectory(
            prefix="persona-buddy-terminal-"
        ) as temporary_value:
            temporary = temporary_value
            isolated = Path(temporary)
            preferences = isolated / "persona_buddy.json"
            staged_captures = isolated / "captures"
            phase = "first:startup"
            first = _run_child(
                root=root,
                preferences=preferences,
                report=isolated / "first.json",
                drive=True,
                phase="first",
                capture_dir=staged_captures,
                inject_child_failure=inject_child_failure,
                inject_startup_noise=inject_startup_noise,
            )
            restored = json.loads(preferences.read_text(encoding="utf-8"))
            restored["open"] = True
            preferences.write_text(
                json.dumps(restored, sort_keys=True), encoding="utf-8"
            )
            phase = "restore:startup"
            second = _run_child(
                root=root,
                preferences=preferences,
                report=isolated / "second.json",
                drive=False,
                phase="restore",
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
                "modal_resume": first["observed"]["modal_resume"],
                "modal_close_replay": first["observed"]["modal_close_replay"],
                "navigation": (
                    first["navigation_count"] == 1
                    and first["screen_generation"] >= 1
                    and first["observed"]["navigation_view"]
                ),
                "viewport_clamp": first["observed"]["viewport_clamp"],
                "compact_controls": first["observed"]["compact_controls"],
                "compact_move_restore": first["observed"]["compact_move_restore"],
                "capture_release": first["capture_released"],
                "paint": first["observed"]["paint"],
                "geometry_restore": second["loaded_geometry"] == first["geometry"],
                "graceful_exit": (
                    first["observed"]["graceful_exit"]
                    and second["observed"]["graceful_exit"]
                ),
                "pet_only_normal": first["observed"]["pet_only_normal"],
                "fixed_alert_replaces_pet": first["observed"][
                    "fixed_alert_replaces_pet"
                ],
                "real_folded_thumbnail": first["observed"]["real_folded_thumbnail"],
                "constrained_two_icons": first["observed"]["constrained_two_icons"],
            }
            if inject_check_failure:
                checks["drag"] = False
            captures_complete = all(
                (staged_captures / name).is_file()
                and (staged_captures / name).stat().st_size > 0
                for name in _CAPTURE_NAMES
            )
            if not captures_complete:
                raise _ProbeChildFailure(
                    category="persona_buddy_terminal_capture_incomplete",
                    phase="parent:capture",
                    child_return_code=0,
                    diagnostic_tail="persona_buddy_terminal_capture_incomplete",
                )
            if not all(checks.values()):
                failed_checks = ",".join(
                    name for name, passed in checks.items() if not passed
                )
                raise _ProbeChildFailure(
                    category="persona_buddy_terminal_check_failure",
                    phase="parent:checks",
                    child_return_code=0,
                    diagnostic_tail=(
                        "persona_buddy_terminal_check_failure\n"
                        f"failed_checks={failed_checks}"
                    ),
                    checks=checks,
                )
            result = {
                "checks": checks,
                "regions": first["observed_regions"],
                "first": first,
                "second": second,
            }
            phase = "parent:report"
            if report_output is not None:
                _atomic_write_json(
                    report_output,
                    result,
                    inject_replace_failure=inject_report_publication_failure,
                )
            phase = "parent:capture"
            if capture_output is not None:
                _publish_capture_group(
                    staged_captures,
                    capture_output,
                    inject_failure=inject_publication_failure,
                )
            print(json.dumps(result, sort_keys=True))
            print("PASS persona_buddy_terminal")
            return 0
    except _ProbeChildFailure as caught_failure:
        failure = caught_failure
        child_return_code = failure.child_return_code
    except OSError as error:
        category = {
            "parent:admission": "persona_buddy_terminal_output_admission",
            "parent:report": "persona_buddy_terminal_report_publish",
            "parent:capture": "persona_buddy_terminal_capture_publish",
        }.get(phase, "persona_buddy_terminal_probe_failure")
        failure = _ProbeChildFailure(
            category=category,
            phase=phase,
            child_return_code=child_return_code,
            diagnostic_tail=f"{type(error).__name__}: {category}",
        )
    _rollback_managed_captures(capture_output)
    result, artifact = _persist_structured_failure(
        failure,
        report_output=report_output,
        root=root,
        temporary=temporary,
    )
    print(json.dumps(result, sort_keys=True))
    raise RuntimeError(f"{failure.category} artifact={artifact}") from failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("preferences", nargs="?", type=_validated_cli_path)
    parser.add_argument("report", nargs="?", type=_validated_cli_path)
    parser.add_argument("--report", dest="parent_report", type=_validated_cli_path)
    parser.add_argument("--capture-dir", type=_validated_cli_path)
    parser.add_argument("--inject-child-failure", action="store_true")
    parser.add_argument(
        "--inject-publication-failure", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inject-report-publication-failure",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inject-startup-noise", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inject-check-failure", action="store_true", help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()
    if arguments.child:
        if arguments.preferences is None or arguments.report is None:
            return 2
        if arguments.inject_startup_noise:
            os.write(
                sys.stderr.fileno(),
                (
                    b"PERSONA_BUDDY_PRIVATE_STARTUP_MARKER "
                    b"/private/tmp/private-checkout/config.toml provider_inventory\n"
                    + b"x"
                    * (_CAPTURE_BYTES * 2)
                ),
            )
        if arguments.inject_child_failure:
            raise RuntimeError("persona_buddy_injected_child_failure")
        return _child(arguments.preferences, arguments.report)
    if arguments.capture_dir is None:
        parser.error("--capture-dir is required")
    try:
        return _parent(
            arguments.parent_report,
            capture_output=arguments.capture_dir,
            inject_child_failure=arguments.inject_child_failure,
            inject_publication_failure=arguments.inject_publication_failure,
            inject_report_publication_failure=(
                arguments.inject_report_publication_failure
            ),
            inject_startup_noise=arguments.inject_startup_noise,
            inject_check_failure=arguments.inject_check_failure,
        )
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
