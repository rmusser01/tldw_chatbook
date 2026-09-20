"""Exact artifact handoffs into Library without owning reads or source data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from textual.widgets import Input

from ...Constants import (
    LIBRARY_NAV_CONTEXT_ARTIFACT_CHATBOOK_ID,
    LIBRARY_NAV_CONTEXT_MODE,
)
from ...Library.library_artifacts_state import ArtifactKey, ArtifactScope
from ..Navigation.pending_handoff_store import (
    ARTIFACT_CHATBOOK_RECORD_PREFIX,
    HandoffChannel,
    HandoffClaim,
    PendingHandoffStore,
)

_CHANNEL = HandoffChannel.ARTIFACT_CHATBOOK_TARGET
_CHATBOOKS_ROW = "artifacts-chatbooks"


@dataclass(frozen=True)
class _TargetRequest:
    claim: HandoffClaim[str]
    key: ArtifactKey
    controller: Any
    generation: int
    profile: tuple[int, ...]


class LibraryArtifactsNavigation:
    """Keep one exact claim until page application or a terminal missing result.

    The screen supplies its ordinary dirty-guarded navigation admission. The
    artifact controller owns the locator worker and calls the settlement hooks
    only after its existing profile/scope/generation fences have passed.
    """

    def __init__(self, screen: Any) -> None:
        self.screen = screen
        self._claim: HandoffClaim[str] | None = None
        self._request: _TargetRequest | None = None
        self._eligible = False
        self._retry_pending = False
        self._route_requested = False

    @property
    def store(self) -> PendingHandoffStore:
        return self.screen.app_instance.pending_handoffs

    def prepare_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize explicit artifact intent without replacing other Library intent.

        Args:
            context: Incoming navigation context before existing save/admission guards.

        Returns:
            A detached context. Exact artifact targets use the Chatbooks canvas;
            unrelated explicit modes and record IDs retain precedence.
        """
        prepared = dict(context)
        mode = prepared.get(LIBRARY_NAV_CONTEXT_MODE)
        explicit = prepared.get(LIBRARY_NAV_CONTEXT_ARTIFACT_CHATBOOK_ID)
        allowed_keys = {
            LIBRARY_NAV_CONTEXT_MODE,
            LIBRARY_NAV_CONTEXT_ARTIFACT_CHATBOOK_ID,
        }
        if (
            mode not in (None, "artifacts-all", _CHATBOOKS_ROW)
            or set(prepared) - allowed_keys
        ):
            self.invalidate()
            return prepared
        if mode is None and explicit is None:
            return prepared
        self._eligible = True
        self._retry_pending = False
        self._route_requested = False
        if explicit is not None:
            self.store.stage(_CHANNEL, explicit)
            self._release()
            prepared.pop(LIBRARY_NAV_CONTEXT_ARTIFACT_CHATBOOK_ID)
        if self.store.has_pending(_CHANNEL) or self._claim is not None:
            prepared[LIBRARY_NAV_CONTEXT_MODE] = _CHATBOOKS_ROW
        return prepared

    def resume(self) -> None:
        """Start one admitted target; route changes still pass the screen's guards."""
        screen = self.screen
        if (
            not self._eligible
            or self._retry_pending
            or not screen.is_mounted
            or screen.app.screen is not screen
        ):
            return
        if self._claim is not None:
            if self.store.is_current_claim(self._claim):
                return
            self._release()
        if not self.store.has_pending(_CHANNEL):
            return
        if screen._library_selected_row_id != _CHATBOOKS_ROW:
            if not self._route_requested:
                self._route_requested = True
                screen.apply_navigation_context(
                    {LIBRARY_NAV_CONTEXT_MODE: _CHATBOOKS_ROW}
                )
            return
        controller = screen._artifacts_controller
        if controller is None:
            return
        claim = self.store.claim(_CHANNEL)
        if claim is None:
            return
        self._claim = claim
        self._route_requested = False
        try:
            native_id = claim.value.removeprefix(ARTIFACT_CHATBOOK_RECORD_PREFIX)
            key = ArtifactKey("chatbook", int(native_id))
            if str(key.native_id) != native_id:
                raise ValueError("Noncanonical Chatbook ID")
        except ValueError:
            if self.store.is_current_claim(claim):
                # No locator was started, so fence a prior generic page worker.
                controller.generation += 1
                controller.target_unavailable()
                self.store.acknowledge_current(claim)
                self._claim = None
            return
        try:
            controller.enter_view("chatbooks")
            # An exact source target takes precedence over restored filters.
            controller.stop_timer()
            controller.scope = ArtifactScope(view="chatbooks")
            if controller.shell is not None and controller.shell.is_mounted:
                field = controller.shell.query_one("#library-artifacts-search", Input)
                with field.prevent(Input.Changed):
                    field.value = ""
            controller.open_target(key)
            self._request = _TargetRequest(
                claim, key, controller, controller.generation, controller.profile()
            )
        except Exception:
            self._release()
            raise

    def target_is_current(self, key: ArtifactKey, generation: int) -> bool:
        """Reject stale claimed targets while leaving ordinary Keep location alone."""
        controller = self.screen._artifacts_controller
        if controller is None or generation != controller.generation:
            return False
        request = self._request
        if request is None or (key, generation) != (request.key, request.generation):
            return True
        return (
            self._claim is request.claim
            and request.controller is controller
            and request.profile == controller.profile()
            and self.store.is_current_claim(request.claim)
        )

    def target_finished(
        self, key: ArtifactKey, generation: int, *, missing: bool = False
    ) -> None:
        """Acknowledge one applied exact page or terminal missing-target recovery."""
        request = self._request
        if request is None or (key, generation) != (request.key, request.generation):
            return
        if self.target_is_current(key, generation) and self.store.acknowledge_current(
            request.claim
        ):
            self._claim = None
            self._retry_pending = False

    def target_failed(self, key: ArtifactKey, generation: int) -> None:
        """Requeue a failed source lookup for explicit Retry, never as missing."""
        request = self._request
        if request is None or (key, generation) != (request.key, request.generation):
            return
        if self.target_is_current(key, generation):
            self._release()
            self._retry_pending = True

    def retry(self) -> bool:
        """Retry the exact failed handoff, returning whether this handled Retry."""
        if not self._retry_pending:
            return False
        self._retry_pending = False
        self._eligible = True
        self.resume()
        return True

    def invalidate(self) -> None:
        """Release abandoned claims without overwriting newer app-owned targets."""
        request = self._request
        if self._claim is not None and request is not None:
            controller = request.controller
            if controller.generation == request.generation:
                controller.generation += 1
                controller.loading = False
        self._release()
        self._eligible = False
        self._retry_pending = False
        self._route_requested = False

    def _release(self) -> None:
        if self._claim is not None:
            claim, self._claim = self._claim, None
            self.store.release(claim)
