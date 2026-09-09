"""App-owned mounting and trusted lifecycle wiring for the local Persona Buddy."""

from __future__ import annotations

import asyncio
import weakref
from typing import Any

from textual.screen import ModalScreen

from .base_app_screen import BaseAppScreen


def buddy_screen_allowed(app: Any, screen: Any) -> bool:
    """Keep the Buddy out of startup, recovery and protected input surfaces."""
    return (
        isinstance(screen, BaseAppScreen)
        and not isinstance(screen, ModalScreen)
        and not getattr(app, "splash_screen_active", False)
        and screen.screen_name not in {"splash", "auth", "login", "recovery"}
    )


def trusted_console_states(runtime: Any) -> dict[str, str]:
    """Read lifecycle metadata only; model text cannot direct the Buddy."""
    result: dict[str, str] = {}
    controller = getattr(runtime, "chat_controller", None)
    store = getattr(runtime, "chat_store", None)
    bridge = getattr(runtime, "agent_bridge", None)
    read_activity = getattr(bridge, "live_snapshot", None)
    if controller is not None and store is not None:
        for session in store.sessions():
            source = f"console:{session.id}"
            run = controller.run_state_for(session.id)
            status = getattr(run.status, "value", run.status)
            if status in {"validating", "streaming", "checking_citations", "retrying"}:
                result[f"{source}:run"] = "thinking"
                if callable(read_activity):
                    # The bridge publishes step kinds immediately before tool
                    # dispatch and after results. Use its current primary slot,
                    # never historical rows, child summaries, or step text.
                    conversation_id = (
                        getattr(session, "persisted_conversation_id", None)
                        or session.id
                    )
                    activity = read_activity(conversation_id)
                    if (
                        activity.status == "running"
                        and activity.steps
                        and activity.steps[-1].kind == "tool_call"
                    ):
                        result[f"{source}:tool"] = "tool_running"
            elif status == "failed":
                result[f"{source}:run"] = "error"
            if controller.has_pending_approval_round(session.id):
                result[f"{source}:approval"] = "approval_needed"
    view = getattr(runtime, "view", None)
    if view is not None:
        if getattr(view, "_console_dictation_state", "idle") == "recording":
            result["console:dictation"] = "listening"
        if getattr(view, "_console_speaking_message_id", None):
            result["console:speech"] = "speaking"
        # The realtime PCM loop keeps one-shot dictation idle. Read each
        # voice loop's trusted FSM, without consulting transcripts or text.
        for attribute, source, states in (
            (
                "_console_realtime",
                "console:realtime",
                {
                    "live": "listening",
                    "thinking": "thinking",
                    "speaking": "speaking",
                },
            ),
            (
                "_console_hands_free",
                "console:hands-free",
                {
                    "listening": "listening",
                    "countdown": "listening",
                    "awaiting_reply": "thinking",
                    "speaking": "speaking",
                },
            ),
        ):
            loop = getattr(getattr(view, attribute, None), "controller", None)
            state = states.get(getattr(loop, "state", None))
            if state is not None:
                result[source] = state
    return result


class PersonaBuddyHost:
    """Own one serialized refresh loop and weak, replaceable screen views."""

    def __init__(self, app: Any, controller: Any) -> None:
        self._app_ref = weakref.ref(app)
        self.controller = controller
        self._view_ref: Any = lambda: None
        self._task: asyncio.Task | None = None
        self._timer: Any = None
        self._closed = False
        self._epoch = 0
        self._signals: dict[str, str] = {}
        self._prepared_key: Any = None
        self._prepared: Any = None

    @property
    def current_view(self) -> Any:
        """Return only the attached view on the currently active safe screen."""
        view = self._view_ref()
        app = self._app_ref()
        if (
            not self._closed
            and view is not None
            and app is not None
            and view.is_attached
            and view.screen is app.screen
            and buddy_screen_allowed(app, app.screen)
        ):
            return view
        return None

    def start(self) -> None:
        """Subscribe to actual screen changes and poll lightweight authority."""
        app = self._app_ref()
        app.screen_change_signal.subscribe(app, self.screen_changed)
        self._timer = app.set_interval(0.75, self.request_refresh)
        self.request_refresh()

    def sync_trusted_signals(self, signals: dict[str, str]) -> None:
        """Renew active operations but lease a terminal failure only once."""
        for source in self._signals.keys() - signals.keys():
            self.controller.release(source)
        for source, state in signals.items():
            if state != "error" or self._signals.get(source) != "error":
                self.controller.signal(
                    source, state, ttl=3.0 if state == "error" else 2.0
                )
        self._signals = dict(signals)

    def screen_changed(self, _screen: Any) -> None:
        """Immediately hide the old view before any asynchronous replacement."""
        self._epoch += 1
        view = self._view_ref()
        if view is not None:
            view.display = False
            view.release_mouse()
        self.request_refresh()

    def request_refresh(self) -> None:
        """Coalesce refresh requests without cancelling a running decoder."""
        if self._closed or (self._task is not None and not self._task.done()):
            return
        self._task = asyncio.create_task(self.refresh(), name="persona-buddy-refresh")

    async def select(self, persona_id: str, *, source: str = "local") -> bool:
        """Apply only an explicit eligible local Persona selection."""
        self._epoch += 1
        previous = self._view_ref()
        if previous is not None:
            previous.display = False
            previous.release_mouse()
        selected = await self.controller.select(persona_id, source=source)
        if selected:
            self._epoch += 1
            self.request_refresh()
        return selected

    async def update_preferences(self, **changes: Any) -> None:
        """Persist window controls through the controller's private store."""
        await self.controller.update_preferences(**changes)
        self._epoch += 1
        self.request_refresh()

    async def refresh(self) -> None:
        """Resolve off-thread and fence every awaited DOM change."""
        app = self._app_ref()
        if app is None or self._closed:
            return
        epoch = self._epoch
        try:
            await self.controller.load_preferences()
            if self._closed or epoch != self._epoch:
                return
            screen = app.screen
            prefs = self.controller.preferences
            view = self._view_ref()
            allowed = buddy_screen_allowed(app, screen)
            if not allowed or not prefs.enabled or not prefs.open:
                if view is not None:
                    view.display = False
                    view.release_mouse()
                return
            signals = trusted_console_states(getattr(app, "console_runtime", None))
            self.sync_trusted_signals(signals)
            viewport = (screen.size.width, screen.size.height)
            width = max(1, min(prefs.width, viewport[0]) - 2)
            height = max(1, min(prefs.height, viewport[1] - 2) - 5)
            from ...config import get_cli_setting
            from ...Widgets.Persona_Widgets.persona_buddy import (
                PersonaBuddyView,
                prepare_buddy_frames,
            )

            reduced = bool(get_cli_setting("appearance", "reduce_motion", False))
            monochrome = bool(getattr(app.console, "no_color", False))

            def prepare(resolution: Any) -> Any:
                key = (resolution.cache_identity, width, height, monochrome)
                if key != self._prepared_key:
                    self._prepared = prepare_buddy_frames(
                        resolution, width, height, monochrome=monochrome
                    )
                    self._prepared_key = key
                return self._prepared

            snapshot = await self.controller.refresh(
                width,
                height,
                reduced_motion=reduced or prefs.collapsed,
                prepare=prepare,
            )
            if (
                self._closed
                or epoch != self._epoch
                or app.screen is not screen
                or viewport != (screen.size.width, screen.size.height)
                or prefs != self.controller.preferences
                or not buddy_screen_allowed(app, screen)
            ):
                return
            if snapshot is None:
                if view is not None:
                    view.display = False
                return
            if snapshot.generation != self.controller.generation:
                return
            if view is None or not view.is_attached or view.screen is not screen:
                if view is not None and view.is_attached:
                    await view.remove()
                    if self._closed or epoch != self._epoch or app.screen is not screen:
                        return
                candidate = PersonaBuddyView(snapshot, prefs)
                await screen.mount(candidate)
                if (
                    self._closed
                    or epoch != self._epoch
                    or app.screen is not screen
                    or snapshot.generation != self.controller.generation
                ):
                    await candidate.remove()
                    return
                view = candidate
                self._view_ref = weakref.ref(view)
            view.apply_preferences(prefs)
            view.apply_snapshot(snapshot)
            view.display = True
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — optional UI must fail path-free.
            # Rendering is optional; never leak private storage paths or break navigation.
            view = self._view_ref()
            if view is not None:
                view.display = False

    async def shutdown(self) -> None:
        """Drain the actual refresh task before the app closes persistence."""
        self._closed = True
        self._epoch += 1
        if self._timer is not None:
            self._timer.stop()
        cancelled = False
        if self._task is not None:
            self._task.cancel()
            while not self._task.done():
                try:
                    await asyncio.shield(self._task)
                except asyncio.CancelledError:
                    if not self._task.done():
                        cancelled = True
        await self.controller.shutdown()
        if cancelled:
            raise asyncio.CancelledError
