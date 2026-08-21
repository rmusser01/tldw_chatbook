"""Generic Console interrupt-round host (sub-project C1, task program spec
2026-08-20-console-interrupt-host-design.md).

One lifecycle, ONE lock, per-kind storage for the Console's blocking
interrupt rounds (MCP approvals, skill-install confirms, skill-script
confirms; "question" reserved for sub-project A).

Locking: ``lock`` is a plain NON-REENTRANT ``threading.Lock``. The
controller aliases its three historical lock names to this one object, so
nesting any two of them -- or calling a host method that takes the lock
from inside a ``with`` on any of the names -- self-deadlocks immediately.
Nothing nests today (verified at C1 design time, tests included); keep it
that way.

Seams: the host holds the CONTROLLER and reads ``.app``, ``.store``, and
the per-kind setter attributes late-bound, at call time -- they are
assigned at screen attach, after construction, and UI tests swap in
controller doubles. The full surface the host may touch is exactly what
``Tests/Chat/test_console_interrupt_rounds.py``'s ``FakeSeams`` provides.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from tldw_chatbook.Agents.human_input_wait import use_human_input_wait

#: Kind -> the controller attribute holding that kind's UI setter. The
#: setters are attach-time assignments and may be absent entirely
#: (headless, or a kind not yet wired -- "question" until sub-project A);
#: every read goes through ``getattr(..., None)`` and treats None as
#: "no UI, no-op".
KIND_SETTER_ATTRS: dict[str, str] = {
    "approval": "set_pending_approval",
    "skill_install": "set_pending_skill_install",
    "skill_script": "set_pending_skill_script",
    "question": "set_pending_question",
}


class InterruptRoundHost:
    """Own the registries, payload maps, and FIFO-head render contract."""

    POLL_SECONDS = 1.0

    def __init__(self, seams: Any) -> None:
        self._seams = seams
        self.lock = threading.Lock()
        self.registries: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }
        self.payloads: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }

    # -- setter / app access (always late-bound) -----------------------

    def _setter(self, kind: str):
        return getattr(self._seams, KIND_SETTER_ATTRS[kind], None)

    def _active_session_id(self) -> str:
        store = getattr(self._seams, "store", None)
        return (getattr(store, "active_session_id", None) or "") if store else ""

    # -- payload layer (moved verbatim from ConsoleChatController) -----

    @staticmethod
    def _head_locked(
        store: dict[str, dict[str, Any]], session_id: str | None
    ) -> dict[str, Any] | None:
        """The session's oldest-armed payload. Caller holds ``lock``."""
        for payload in store.values():
            if payload.get("session_id") == session_id:
                return payload
        return None

    def park_round_payload(
        self, kind: str, round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head."""
        session_id = payload.get("session_id")
        store = self.payloads[kind]
        with self.lock:
            store[round_id] = payload
            head = self._head_locked(store, session_id)
        return head is payload

    def head_round_payload(
        self, kind: str, session_id: str
    ) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should currently show.

        Carries the PR #1836 remaining-time snapshot behavior verbatim:
        a payload with a live ``deadline_monotonic`` is returned as a
        shallow copy whose ``timeout_seconds`` is the remaining window;
        the retained payload is never mutated. The ``head is payload``
        identity check in ``park_round_payload`` goes through
        ``_head_locked`` and is unaffected.
        """
        store = self.payloads[kind]
        with self.lock:
            payload = self._head_locked(store, session_id)
        if payload is None:
            return None
        deadline = payload.get("deadline_monotonic")
        if not deadline:
            return payload
        snapshot = dict(payload)
        snapshot["timeout_seconds"] = max(0.0, deadline - time.monotonic())
        return snapshot

    def session_round_payloads(
        self, kind: str, session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload ``kind`` retains for ``session_id``, arm order."""
        store = self.payloads[kind]
        with self.lock:
            return [
                payload
                for payload in store.values()
                if payload.get("session_id") == session_id
            ]

    def unpark_round_payload(self, kind: str, round_id: str) -> None:
        """Drop one round's retained payload. Idempotent."""
        with self.lock:
            self.payloads[kind].pop(round_id, None)

    def remount_head(self, kind: str, session_id: str | None) -> None:
        """Enqueue a head re-derive onto the UI thread (worker-safe).

        The decision -- WHICH payload, and whether the session is still
        the one being viewed -- is computed INSIDE the callable, on the
        UI thread, never from a worker-thread snapshot: the invariant
        three pre-PR0 fix rounds converged on.

        ``session_id=None`` means "the session being VIEWED when the
        callback runs" (legacy no-session rounds mount unconditionally,
        so their card can sit over any session by teardown time).
        """
        app = getattr(self._seams, "app", None)
        if app is None or self._setter(kind) is None:
            return

        def _apply() -> None:
            setter = self._setter(kind)
            if setter is None:
                return
            if session_id is None:
                setter(self.head_round_payload(kind, self._active_session_id()))
                return
            if session_id != self._active_session_id():
                return
            setter(self.head_round_payload(kind, session_id))

        app.call_from_thread(_apply)

    # -- generic round lifecycle ----------------------------------------

    def run_round(
        self,
        kind: str,
        round_id: str,
        payload: dict[str, Any],
        state: dict[str, Any],
        *,
        session_id: str | None,
        owning_session_id: str,
        deadline: float | None,
        is_parked: bool,
        announce_detached: Callable[[], None] | None = None,
        human_wait_run_id: str | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_timeout: Callable[[], None] | None = None,
        check_revoked: bool = True,
    ) -> str:
        """One blocking interrupt round, registration through teardown.

        WORKER THREAD. Reproduces the (converged) bridge lifecycle:
        register -> badge -> park -> announce/park-toast/mount -> poll ->
        teardown (pop, unpark, badge-discard, head re-derive). The
        per-bridge deltas ride the hooks: ``announce_detached`` is the
        MCP detached-view leg; ``on_cancelled``/``on_timeout`` let the
        approvals wrapper stamp its decisions box and audit-log;
        ``human_wait_run_id`` wraps the wait in ``use_human_input_wait``
        (all three bridges wrap theirs today -- ``None`` selects
        ``nullcontext()`` instead, for a kind/test with no owning run to
        pause); ``check_revoked`` is False for skill-install, which is
        never swept.
        """
        event: threading.Event = state["event"]
        with self.lock:
            self.registries[kind][round_id] = state
        is_head = True
        if session_id is not None:
            add = getattr(self._seams, "add_pending_round", None)
            if add is not None:
                add(session_id, round_id)
            is_head = self.park_round_payload(kind, round_id, payload)
        try:
            app = getattr(self._seams, "app", None)
            park_toast = getattr(self._seams, "park_pending_approval", None)
            if announce_detached is not None:
                announce_detached()
            elif is_parked:
                if app is not None and park_toast is not None:
                    app.call_from_thread(park_toast, session_id)
            elif is_head:
                setter = self._setter(kind)
                if app is not None and setter is not None:
                    app.call_from_thread(setter, payload)
            outcome = "decided"
            wait_cm = (
                use_human_input_wait(human_wait_run_id)
                if human_wait_run_id is not None
                else nullcontext()
            )
            with wait_cm:
                while not event.wait(self.POLL_SECONDS):
                    if self._seams._is_session_cancelled(
                        session_id,
                        cancel_event=state.get("cancel_event"),
                        visit_event=state.get("visit_event"),
                    ):
                        if on_cancelled is not None:
                            on_cancelled()
                        outcome = "cancelled"
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        if on_timeout is not None:
                            on_timeout()
                        outcome = "timeout"
                        break
            if check_revoked:
                with self.lock:
                    if bool(state.get("revoked")):
                        outcome = "revoked"
            return outcome
        finally:
            with self.lock:
                self.registries[kind].pop(round_id, None)
            self.unpark_round_payload(kind, round_id)
            if session_id is not None:
                discard = getattr(self._seams, "discard_pending_round", None)
                if discard is not None:
                    discard(session_id, round_id)
            try:
                self.remount_head(
                    kind, owning_session_id if session_id is not None else None
                )
            except Exception:  # noqa: BLE001 -- teardown must never raise
                pass
