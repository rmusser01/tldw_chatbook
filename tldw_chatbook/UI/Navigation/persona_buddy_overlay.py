"""App-owned lifetime for disposable Persona Buddy views.

This module stays lightweight while Buddy is disabled. Rendering imports belong
only to the branch that mounts an enabled view.
"""

from __future__ import annotations

import asyncio
from typing import Any

from textual.message import Message


class PersonaBuddyChanged(Message):
    """Notify the app of a controller generation change without carrying content."""


class PersonaBuddyOverlay:
    """Own one presentation generation across primary screens and rebuilds."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.view: Any = None
        self.screen: Any = None
        self.generation = 0
        self._lock = asyncio.Lock()
        self._worker: Any = None
        self._pending = False
        self.closed = False

    def is_current(self, view: Any) -> bool:
        """Return whether a view may interact with the active primary screen."""
        return bool(
            not self.closed
            and view is self.view
            and view.view_generation == self.generation
            and view.is_attached
            and self.screen is self.app.screen
            and self.screen.is_attached
        )

    def request(self) -> None:
        """Coalesce changes without cancelling mounts or geometry writes."""
        if self.closed:
            return
        self._pending = True
        if self._worker is None or self._worker.is_finished:
            self._worker = self.app.run_worker(
                self._run_pending,
                group="persona-buddy-view-reconcile",
                exclusive=True,
            )

    async def _run_pending(self) -> None:
        while self._pending and not self.closed:
            self._pending = False
            await self.reconcile()

    def _sync_affordances(self, screen: Any) -> None:
        if screen is self.app.screen and screen.is_attached:
            sync = getattr(screen, "sync_persona_buddy_reconciled_state", None)
            if callable(sync):
                sync()

    async def flush_geometry(self) -> None:
        """Drain geometry before the controller closes write admission."""
        if self.view is not None:
            await self.view.flush_pending_geometry_persist()

    async def _retire(self) -> None:
        current = self.view
        self.generation += 1
        if current is None:
            return
        current.release_interaction_capture()
        await current.flush_pending_geometry_persist()
        if current.is_attached:
            await current.remove()
        if self.view is current:
            self.view = None
            self.screen = None

    async def reconcile(self) -> bool:
        """Reconcile current authority; return whether the current view is absent."""
        from .base_app_screen import BaseAppScreen

        async with self._lock:
            if self.closed:
                return True
            screen = self.app.screen
            controller = getattr(self.app, "persona_buddy_controller", None)
            snapshot = controller.snapshot() if controller is not None else None
            desired = bool(
                snapshot is not None
                and snapshot.enabled
                and snapshot.open
                and snapshot.selection is not None
                and not self.app.is_persona_buddy_confirmed_unavailable(
                    controller, snapshot
                )
            )
            if not desired:
                await self._retire()
                self._sync_affordances(screen)
                return True
            if not isinstance(screen, BaseAppScreen) or not screen.is_active:
                # A modal covers the retained primary view. Its is_current fence
                # suspends interaction and resolution until the primary resumes.
                return False
            if not screen.is_attached:
                return False
            if self.view is not None and (
                self.screen is not screen
                or not self.view.is_attached
                or self.view.view_generation != self.generation
                or self.view._controller is not controller
            ):
                await self._retire()
            # Geometry flushing / removal can yield to shutdown or navigation.
            if self.closed:
                return True
            if screen is not self.app.screen or not screen.is_attached:
                self._pending = True
                return True
            latest = controller.snapshot()
            if latest != snapshot:
                self._pending = True
                return self.view is None
            if self.view is not None:
                self.view.refresh_from_controller()
                self.view.resume_resolution()
                self._sync_affordances(screen)
                return False

            from ...Widgets.Persona_Widgets.persona_buddy_widget import (
                PersonaBuddyWidget,
            )

            self.generation += 1
            generation = self.generation
            view = PersonaBuddyWidget(
                controller=controller,
                view_generation=generation,
                reconcile=self.app.reconcile_persona_buddy_view,
                is_current=self.is_current,
                confirm_unavailable=lambda **kwargs: (
                    self.app.confirm_persona_buddy_unavailable(
                        screen=screen, view_generation=generation, **kwargs
                    )
                ),
                is_confirmed_unavailable=self.app.is_persona_buddy_confirmed_unavailable,
            )
            self.view = view
            self.screen = screen
            try:
                await screen.mount(view)
                current = controller.snapshot()
                valid = bool(
                    self.is_current(view)
                    and controller
                    is getattr(self.app, "persona_buddy_controller", None)
                    and current.enabled
                    and current.open
                    and current.selection is not None
                )
                if not valid:
                    view.release_interaction_capture()
                    if view.is_attached:
                        await view.remove()
                    if self.view is view:
                        self.view = None
                        self.screen = None
                    return True
            except BaseException:
                if self.view is view:
                    self.view = None
                    self.screen = None
                view.release_interaction_capture()
                if view.is_attached:
                    await view.remove()
                raise
            self._sync_affordances(screen)
            return False

    async def shutdown(self) -> None:
        """Close presentation admission and drain geometry before domain shutdown."""
        self.closed = True
        self._pending = False
        async with self._lock:
            await self.flush_geometry()
