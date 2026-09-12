"""Generic Console interrupt-round host (sub-project C1, task program spec
2026-08-20-console-interrupt-host-design.md).

One lifecycle, ONE lock, per-kind storage for the Console's five blocking
interrupt rounds: MCP approvals, skill-install confirms, skill-script
confirms, worktree-merge confirms, and ask_user questions (task-31384,
sub-project C of the 2026-08-19 design; the C1 spine from PR #1903 ported
to the five-kind controller).

Locking: ``lock`` is a plain NON-REENTRANT ``threading.Lock``. The
controller aliases its five historical lock names to this one object, so
nesting any two of them -- or calling a host method that takes the lock
from inside a ``with`` on any of the names -- self-deadlocks immediately.
Nothing nests today (verified at C1 design time, tests included); keep it
that way.

Seams: the host holds the CONTROLLER and reads everything off it
late-bound, at call time -- ``.app``, ``.store``, and the per-kind setter
attributes are assigned at screen attach, after construction, and UI
tests swap in controller doubles. The surface splits by entry point:

* the payload helpers and ``remount_head`` touch only ``.app``,
  ``.store``, and the ``KIND_SETTER_ATTRS`` setters -- that is exactly
  what ``Tests/Chat/test_console_interrupt_rounds.py``'s ``FakeSeams``
  provides;
* ``run_round`` additionally calls ``_is_session_cancelled`` (its poll
  probe), ``add_pending_round``/``discard_pending_round`` (the fleet
  badge), and reads ``park_pending_approval`` (the background-session
  toast) -- the superset that file's ``FakeSeamsFull`` provides.

A double built for ``run_round`` therefore needs ``FakeSeamsFull``'s
surface, not ``FakeSeams``'.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_chatbook.Agents.human_input_wait import use_human_input_wait

#: Kind -> the controller attribute holding that kind's UI setter. The
#: setters are attach-time assignments and may be absent entirely
#: (headless, or a kind not yet wired -- "question" until sub-project A);
#: every read goes through ``getattr(..., None)`` and treats None as
#: "no UI, no-op".
#: The kinds every session-activation site re-derives together
#: (`remount_for_session`); approvals are re-derived separately by the
#: sites' own approval block, which also drives the attach path.
SESSION_REMOUNT_KINDS: tuple[str, ...] = (
    "skill_install",
    "skill_script",
    "worktree_merge",
    "question",
)

KIND_SETTER_ATTRS: dict[str, str] = {
    "approval": "set_pending_approval",
    "skill_install": "set_pending_skill_install",
    "skill_script": "set_pending_skill_script",
    "worktree_merge": "set_pending_worktree_merge",
    "question": "set_pending_question",
}


def head_payload_locked(
    store: dict[str, dict[str, Any]], session_id: str | None
) -> dict[str, Any] | None:
    """The session's oldest-armed payload in ``store``. Caller holds the lock.

    Args:
        store: A round-id -> payload dict.
        session_id: The session to look up.

    Returns:
        The first payload armed for ``session_id``, or None.
    """
    for payload in store.values():
        if payload.get("session_id") == session_id:
            return payload
    return None


def park_round_payload(
    lock: threading.Lock,
    store: dict[str, dict[str, Any]],
    round_id: str,
    payload: dict[str, Any],
) -> bool:
    """Retain ``payload`` under ``round_id``.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        round_id: The round's id.
        payload: The card payload; must carry ``session_id``.

    Returns:
        True when ``payload`` is now its session's head.
    """
    with lock:
        store[round_id] = payload
        head = head_payload_locked(store, payload.get("session_id"))
    return head is payload


def head_round_payload(
    lock: threading.Lock, store: dict[str, dict[str, Any]], session_id: str
) -> dict[str, Any] | None:
    """The payload whose card ``session_id`` should show now.

    Carries the PR #1836 remaining-time snapshot behaviour verbatim: a
    payload with a live ``deadline_monotonic`` is returned as a shallow
    copy whose ``timeout_seconds`` is the remaining window; the retained
    payload is never mutated.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        session_id: The session whose head to return.

    Returns:
        The head payload (or its remaining-time snapshot), or None.
    """
    with lock:
        payload = head_payload_locked(store, session_id)
    if payload is None:
        return None
    deadline = payload.get("deadline_monotonic")
    if not deadline:
        return payload
    snapshot = dict(payload)
    snapshot["timeout_seconds"] = max(0.0, deadline - time.monotonic())
    return snapshot


def session_round_payloads(
    lock: threading.Lock, store: dict[str, dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    """Every payload ``store`` retains for ``session_id``, arm order first.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        session_id: The session to collect.

    Returns:
        The session's payloads in arm order.
    """
    with lock:
        return [
            payload
            for payload in store.values()
            if payload.get("session_id") == session_id
        ]


def unpark_round_payload(
    lock: threading.Lock, store: dict[str, dict[str, Any]], round_id: str
) -> None:
    """Drop ``round_id``'s retained payload, if any.

    Args:
        lock: The lock guarding ``store``.
        store: A round-id -> payload dict.
        round_id: The round to forget.
    """
    with lock:
        store.pop(round_id, None)


@dataclass
class _DecisionClock:
    """Volatile answerable-time budget; never included in a card payload."""

    remaining: float
    updated_at: float
    payload: dict[str, Any]
    answerable: bool = False
    requires_head: bool = True

    def update(self, now: float, answerable: bool) -> None:
        if self.answerable:
            self.remaining = max(0.0, self.remaining - (now - self.updated_at))
        self.updated_at = now
        self.answerable = answerable
        self.payload["timeout_seconds"] = self.remaining
        self.payload["deadline_monotonic"] = (
            now + self.remaining if answerable else None
        )


class InterruptRoundHost:
    """Own the registries, payload maps, and FIFO-head render contract."""

    POLL_SECONDS = 1.0

    def __init__(self, seams: Any) -> None:
        self._seams = seams
        self.lock = threading.Lock()
        self.registries: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }
        # Host-lifetime fences: logical completion cannot prove that an
        # abandoned provider daemon will never reach its approval fallback.
        self._revoked_runs: dict[str, set[str]] = {
            kind: set() for kind in KIND_SETTER_ATTRS
        }
        #: Per-kind hook run on the UI thread after a head payload is pushed,
        #: whichever path pushed it (teardown promotion, revocation sweep,
        #: activation, attach). Approvals register their ADR-090 permission
        #: summary trigger here once, so no remount path can forget it.
        self.after_remount: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.payloads: dict[str, dict[str, dict[str, Any]]] = {
            kind: {} for kind in KIND_SETTER_ATTRS
        }
        # None preserves the setter-based contract for non-Textual callers.
        # A reused screen explicitly reports suspend/resume, including modals.
        self.view_visible: bool | None = None
        self.decision_view_revision = 0
        self._decision_views: dict[
            object,
            tuple[
                str,
                frozenset[str],
                tuple[weakref.ReferenceType[Any], int, bool, str] | None,
            ],
        ] = {}
        self._retained_decision_targets: dict[
            str, tuple[weakref.ReferenceType[Any], int, bool]
        ] = {}

    def retain_decision_target(self, session_id: str) -> None:
        """An explicit Buddy opener makes this exact session's cards recoverable."""
        session = next(
            (row for row in self._seams.store.sessions() if row.id == session_id), None
        )
        if session is not None:
            with self.lock:
                self._retained_decision_targets[session_id] = (
                    weakref.ref(session),
                    session.conversation_binding_revision,
                    session.ephemeral,
                )

    def has_retained_decision_target(self, session_id: str | None) -> bool:
        """Wake-only, closed, replaced or repurposed slots gain no Buddy capability."""
        with self.lock:
            retained = self._retained_decision_targets.get(session_id)
        if retained is None:
            return False
        reference, revision, ephemeral = retained
        session = reference()
        return bool(
            session is not None
            and session.runtime_backend == "local"
            and session.conversation_binding_revision == revision
            and session.ephemeral == ephemeral
            and any(row is session for row in self._seams.store.sessions())
        )

    def set_decision_view(
        self,
        owner: object,
        session_id: str | None,
        *,
        kinds: tuple[str, ...] = tuple(KIND_SETTER_ATTRS),
        decision_id: str | None = None,
    ) -> None:
        """Claim/release a visible exact-session decision projection, such as Buddy."""
        session = (
            next(
                (row for row in self._seams.store.sessions() if row.id == session_id),
                None,
            )
            if decision_id is not None
            else None
        )
        rendered = (
            (
                weakref.ref(session),
                session.conversation_binding_revision,
                session.ephemeral,
                decision_id,
            )
            if session is not None
            else None
        )
        with self.lock:
            previous = self._decision_views.get(owner)
            self.decision_view_revision += 1
            if session_id is None:
                self._decision_views.pop(owner, None)
            else:
                self._decision_views[owner] = (session_id, frozenset(kinds), rendered)
        self.refresh_decision_clocks()
        refresh = getattr(self._seams, "_refresh_answerable_decision", None)
        if callable(refresh):
            for target in {session_id, previous[0] if previous else None} - {None}:
                refresh(target)

    def rendered_decision_ids(self, session_id: str) -> set[str]:
        """Snapshot live exact-binding card claims without retaining the host lock."""
        with self.lock:
            views = tuple(self._decision_views.values())
        sessions = self._seams.store.sessions()
        result = set()
        for target, _kinds, rendered in views:
            if target != session_id or rendered is None:
                continue
            reference, revision, ephemeral, decision_id = rendered
            session = reference()
            if (
                session is not None
                and session.conversation_binding_revision == revision
                and session.ephemeral == ephemeral
                and any(row is session for row in sessions)
            ):
                result.add(decision_id)
        return result

    def set_view_visible(self, visible: bool) -> None:
        """Account for the old visibility interval before changing clock activity."""
        self.view_visible = visible
        self.refresh_decision_clocks()
        if not visible:
            self.announce_hidden_decisions()

    def refresh_decision_clocks(self) -> None:
        """Charge only a visible session's FIFO head, across every decision kind."""
        with self.lock:
            active_session_id = self._active_session_id()
            now = time.monotonic()
            for kind, states in self.registries.items():
                for state in states.values():
                    clock = state.get("decision_clock")
                    if not isinstance(clock, _DecisionClock):
                        continue
                    session_id = state.get("session_id")
                    head = head_payload_locked(self.payloads[kind], session_id)
                    answerable = (
                        (
                            self.view_visible is not False
                            and self._setter(kind) is not None
                            and (not session_id or session_id == active_session_id)
                        )
                        or any(
                            session_id == target and kind in kinds
                            for target, kinds, _rendered in self._decision_views.values()
                        )
                    ) and (not clock.requires_head or head is clock.payload)
                    clock.update(now, answerable)

    def announce_hidden_decisions(self) -> None:
        """Notify once per live round, including one hidden after it was mounted."""
        announce = getattr(self._seams, "announce_hidden_decision", None)
        if not callable(announce):
            return
        with self.lock:
            pending = []
            typed_pending = []
            for kind, states in self.registries.items():
                for state in states.values():
                    if state.get("decision_type"):
                        typed_pending.append(
                            (
                                kind,
                                state.get("session_id", ""),
                                state.get("decision_id", ""),
                            )
                        )
                        continue
                    if state.get("attention_announced") or state.get("revoked"):
                        continue
                    state["attention_announced"] = True
                    pending.append((state.get("session_id", ""), kind))
        for session_id, kind in pending:
            announce(session_id, kind)
        for kind, session_id, decision_id in typed_pending:
            self._seams._announce_hidden_decision(kind, session_id, decision_id)

    # -- setter / app access (always late-bound) -----------------------

    def _setter(self, kind: str):
        return getattr(self._seams, KIND_SETTER_ATTRS[kind], None)

    def _active_session_id(self) -> str:
        store = getattr(self._seams, "store", None)
        return (getattr(store, "active_session_id", None) or "") if store else ""

    # -- payload layer (moved verbatim from ConsoleChatController) -----

    def park_round_payload(
        self, kind: str, round_id: str, payload: dict[str, Any]
    ) -> bool:
        """Retain ``payload``; return whether it is now its session's head."""
        return park_round_payload(self.lock, self.payloads[kind], round_id, payload)

    def head_round_payload(self, kind: str, session_id: str) -> dict[str, Any] | None:
        """The payload whose card ``session_id`` should show (remaining-time snapshot)."""
        return head_round_payload(self.lock, self.payloads[kind], session_id)

    def session_round_payloads(
        self, kind: str, session_id: str
    ) -> list[dict[str, Any]]:
        """Every payload the kind retains for ``session_id``, arm order first."""
        return session_round_payloads(self.lock, self.payloads[kind], session_id)

    def unpark_round_payload(self, kind: str, round_id: str) -> None:
        """Drop ``round_id``'s retained payload, if any."""
        unpark_round_payload(self.lock, self.payloads[kind], round_id)

    def remount_head(
        self,
        kind: str,
        session_id: str | None,
        *,
        after: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Enqueue a head re-derive onto the UI thread (worker-safe).

        The decision -- WHICH payload, and whether the session is still
        the one being viewed -- is computed INSIDE the callable, on the
        UI thread, never from a worker-thread snapshot: the invariant
        three pre-PR0 fix rounds converged on.

        ``session_id=None`` means "the session being VIEWED when the
        callback runs" (legacy no-session rounds mount unconditionally,
        so their card can sit over any session by teardown time).

        Args:
            kind: The round kind whose head to re-derive.
            session_id: The session to re-derive for, or None for the
                session viewed when the callback runs.
            after: task-31384: a hook run on the UI thread with the pushed
                payload, only when a head was pushed (the approvals bridge
                fires its permission summary here).
        """
        app = getattr(self._seams, "app", None)
        if app is None or self._setter(kind) is None:
            return

        def _apply() -> None:
            setter = self._setter(kind)
            if setter is None:
                return
            target = session_id
            if target is None:
                target = self._active_session_id()
            elif target != self._active_session_id():
                return
            self.refresh_decision_clocks()
            payload = self.head_round_payload(kind, target)
            setter(payload)
            hook = after if after is not None else self.after_remount.get(kind)
            if hook is not None and isinstance(payload, dict):
                hook(payload)

        app.call_from_thread(_apply)

    def pending_total(self) -> int:
        """How many rounds of every kind are registered right now.

        Returns:
            The number of registered interrupt rounds across all kinds.
        """
        with self.lock:
            return sum(len(rounds) for rounds in self.registries.values())

    def _note_pending(self, kind: str, *, raised: bool) -> None:
        """task-31385: tell the seams the pending-round total changed.

        Late-bound like every other seam: a controller (or test double)
        without ``on_pending_rounds_changed`` hears nothing. ``raised`` is
        True right after a round mounted or parked and False after its
        registry entry was popped in teardown.

        Args:
            kind: The round kind that changed.
            raised: Whether this is the arm (True) or the teardown (False).
        """
        hook = getattr(self._seams, "on_pending_rounds_changed", None)
        if hook is None:
            return
        try:
            hook(self.pending_total(), kind, raised)
        except Exception:  # noqa: BLE001 -- attention is best-effort; the round and its teardown are not
            logger.opt(exception=True).debug(
                f"Pending-round attention hook failed for {kind}"
            )

    def register_round(
        self,
        kind: str,
        round_id: str,
        state: dict[str, Any],
        *,
        check_revoked: bool = True,
    ) -> bool:
        """Admit a round atomically with its owner's revocation fence.

        Shared by early controller admission and the host lifecycle. A refused
        preregistration is removed only when it belongs to this exact state.
        """
        with self.lock:
            registry = self.registries[kind]
            run_id = state.get("run_id")
            if check_revoked and (
                state.get("revoked") or run_id in self._revoked_runs[kind]
            ):
                state["revoked"] = True
                if registry.get(round_id) is state:
                    registry.pop(round_id)
                return False
            warn_unowned = (
                check_revoked and not run_id and registry.get(round_id) is not state
            )
            registry[round_id] = state
        if warn_unowned:
            logger.warning("Arming a revocable interrupt round without a run owner")
        return True

    def revoke_for_run(
        self,
        run_id: str,
        stamps: dict[str, Callable[[dict[str, Any]], None]],
    ) -> dict[str, list[tuple[str, str | None]]]:
        """Fence future arms and fail armed rounds owned by ``run_id`` closed.

        Each swept round is marked ``revoked``, stamped closed by its
        kind's callable (approvals deny every undecided key; a skill
        script clears allow/remember; a question needs nothing), removed
        from its registry, unparked, and released via its Event -- the
        controller then discards the fleet badge and re-derives each
        affected session's head, exactly as the per-kind sweeps did.

        Args:
            run_id: The cancelled/abandoned run. Falsy ids sweep nothing.
            stamps: ``kind -> stamp(state)`` for every kind that is swept.
                A kind absent from the map is not swept (skill-install and
                worktree-merge are primary-agent-only and never are).

        Returns:
            ``kind -> [(round_id, session_id), ...]`` for the swept rounds.
        """
        swept: dict[str, list[tuple[str, str | None]]] = {kind: [] for kind in stamps}
        if not run_id:
            return swept
        with self.lock:
            for kind, stamp in stamps.items():
                self._revoked_runs[kind].add(run_id)
                registry = self.registries[kind]
                for round_id, state in list(registry.items()):
                    if state.get("run_id") != run_id:
                        continue
                    state["revoked"] = True
                    stamp(state)
                    registry.pop(round_id, None)
                    swept[kind].append((round_id, state.get("session_id") or None))
                    event = state.get("event")
                    if event is not None:
                        event.set()
        for kind, rounds in swept.items():
            for round_id, _session_id in rounds:
                self.unpark_round_payload(kind, round_id)
        return swept

    def remount_for_session(
        self,
        session_id: str,
        *,
        after: dict[str, Callable[[dict[str, Any]], None]] | None = None,
        kinds: tuple[str, ...] | None = None,
    ) -> None:
        """UI THREAD: push every kind's head for ``session_id`` in one call.

        Args:
            session_id: The session now being activated/viewed.
            after: Optional ``kind -> hook(payload)`` run after that kind's
                setter when a head payload was pushed (task-31384: the
                approvals bridge fires its permission summary here).
            kinds: The kinds to push; every kind when None.
        """
        for kind in kinds if kinds is not None else tuple(KIND_SETTER_ATTRS):
            setter = self._setter(kind)
            if setter is None:
                continue
            payload = self.head_round_payload(kind, session_id)
            setter(payload)
            hook = (after or {}).get(kind)
            if hook is not None and isinstance(payload, dict):
                hook(payload)

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
        announce_detached: Callable[[], bool] | None = None,
        human_wait_run_id: str | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_timeout: Callable[[], None] | None = None,
        check_revoked: bool = True,
        on_teardown: Callable[[], bool] | None = None,
        before_wait: Callable[[], None] | None = None,
        on_outcome: Callable[[str], None] | None = None,
    ) -> str:
        """One blocking interrupt round, registration through teardown.

        WORKER THREAD. Reproduces the (converged) bridge lifecycle:
        register -> badge -> park -> announce/park-toast/mount -> poll ->
        teardown (pop, unpark, badge-discard, head re-derive). The
        per-bridge deltas ride the hooks: ``announce_detached`` is the
        MCP detached-view leg, called after the park and returning True
        when it announced app-wide (the card is then not mounted);
        ``on_cancelled``/``on_timeout`` let the approvals wrapper stamp
        its decisions box and audit-log; ``human_wait_run_id`` wraps the
        wait in ``use_human_input_wait`` (every bridge wraps its wait
        today -- ``None`` selects ``nullcontext()`` instead, for a
        kind/test with no owning run to pause); ``check_revoked`` is False
        for skill-install and worktree-merge, which are never swept;
        ``on_outcome`` runs with the final outcome AFTER the wait and
        BEFORE teardown, so a wrapper can snapshot its decisions, write
        its audit rows or transcript marker while the round is still
        registered and its card still mounted; ``on_teardown``
        (task-31384) lets a kind retain its payload past teardown by
        returning True; ``before_wait`` runs once inside the wait mark
        before polling begins. The teardown head re-derive applies the
        kind's ``after_remount`` hook, like every other remount path.

        Args:
            kind: The round kind (a ``KIND_SETTER_ATTRS`` key).
            round_id: The round's unique id.
            payload: The card payload to park and mount.
            state: The registry entry; must carry an ``event`` and may
                carry ``cancel_event``/``visit_event``/``revoked``.
            session_id: The owning session for badge and park bookkeeping,
                or None for the legacy no-session shape.
            owning_session_id: The session whose head is re-derived at
                teardown.
            deadline: Initial ``time.monotonic()`` deadline, or None to wait
                indefinitely. Finite budgets advance only while the owning
                session's FIFO head can be answered on the visible Console.
            is_parked: True when the round belongs to a non-viewed session.
            announce_detached: Detached-view announcer; returns True when
                it announced instead of mounting.
            human_wait_run_id: Run id for ``use_human_input_wait``, or None.
            on_cancelled: Runs once when the session cancel fires mid-wait.
            on_timeout: Runs once when the answerable-time budget is exhausted.
            check_revoked: Whether a ``revoked`` stamp wins over "decided".
            on_teardown: Returns True to keep the payload parked.
            before_wait: Runs once inside the wait mark before polling.
            on_outcome: Receives the outcome before teardown.

        Returns:
            ``"decided"``, ``"cancelled"``, ``"timeout"`` or ``"revoked"``.
        """
        event: threading.Event = state["event"]
        if not self.register_round(kind, round_id, state, check_revoked=check_revoked):
            if on_outcome is not None:
                on_outcome("revoked")
            return "revoked"
        is_head = True
        publish_decision = getattr(self._seams, "_publish_pending_decision", None)
        retained_decision = (
            session_id is not None
            and kind in {"approval", "skill_install", "skill_script"}
            and callable(publish_decision)
        )
        if retained_decision:
            # The controller's accepted-time metadata uses this same lock.
            # Enter the hook only after releasing the registration lock.
            is_head = publish_decision(
                round_state=state,
                payload=payload,
                decision_type=kind,
                decision_id=round_id,
                timeout_seconds=float(payload.get("timeout_seconds") or 0),
                retained_store=self.payloads[kind],
            )
            deadline = None
        clock = None
        if deadline is not None:
            now = time.monotonic()
            clock = _DecisionClock(
                max(0.0, float(payload.get("timeout_seconds", deadline - now))),
                now,
                payload,
                requires_head=session_id is not None,
            )
            with self.lock:
                state["decision_clock"] = clock
        if session_id is not None:
            add = getattr(self._seams, "add_pending_round", None)
            if add is not None:
                # Qodo #4: the badge does not care which kind is waiting, but
                # the run chip and activity line do -- passing it here is
                # what lets them say "Waiting for your answer" for a question
                # instead of claiming an approval is pending. Optional by
                # keyword so the many two-argument controller doubles keep
                # working; `TypeError` means an older seam, not a bug.
                try:
                    add(session_id, round_id, kind=kind)
                except TypeError:
                    add(session_id, round_id)
            if not retained_decision:
                is_head = self.park_round_payload(kind, round_id, payload)
        try:
            app = getattr(self._seams, "app", None)
            park_toast = getattr(self._seams, "park_pending_approval", None)
            if retained_decision:
                if self._seams._approval_view_is_detached() or is_parked or not is_head:
                    self._seams._announce_hidden_decision(
                        kind, owning_session_id, round_id
                    )
                elif getattr(self._seams, "set_pending_decision", None) is not None:
                    self._seams._marshal_pending_decision_projection()
                else:
                    setter = self._setter(kind)
                    if app is not None and setter is not None:
                        app.call_from_thread(setter, payload)
            elif announce_detached is not None and announce_detached():
                with self.lock:
                    state["attention_announced"] = True
            elif self.view_visible is False:
                self.announce_hidden_decisions()
            elif is_parked:
                if app is not None and park_toast is not None:
                    app.call_from_thread(park_toast, session_id)
            elif is_head:
                setter = self._setter(kind)
                if app is not None and setter is not None:
                    app.call_from_thread(setter, payload)
            self.refresh_decision_clocks()
            # A sweep can revoke and pop the round between registration and
            # here; a round that is no longer registered never announces
            # itself (no bell for a dead round, no zero-total "raised").
            with self.lock:
                still_live = self.registries[kind].get(round_id) is state and not bool(
                    state.get("revoked")
                )
            if still_live:
                self._note_pending(kind, raised=True)
            outcome = "decided"
            wait_cm = (
                use_human_input_wait(human_wait_run_id)
                if human_wait_run_id is not None
                else nullcontext()
            )
            with wait_cm:
                # task-31384: the approvals bridge fires its advisory
                # permission summary INSIDE the human-wait mark so the
                # summariser's own model call never counts against the
                # owning run's tool clock.
                if before_wait is not None:
                    before_wait()
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
                    if retained_decision:
                        self._seams.expire_pending_decisions()
                    self.refresh_decision_clocks()
                    if clock is not None and clock.remaining <= 0:
                        if on_timeout is not None:
                            on_timeout()
                        outcome = "timeout"
                        break
            if retained_decision and state.get("terminal_reason") == "timeout":
                outcome = "timeout"
                if on_timeout is not None:
                    on_timeout()
            if check_revoked:
                with self.lock:
                    if bool(state.get("revoked")):
                        outcome = "revoked"
            if on_outcome is not None:
                on_outcome(outcome)
            return outcome
        finally:
            with self.lock:
                self.registries[kind].pop(round_id, None)
            if retained_decision:
                self._seams._forget_hidden_decision(round_id)
            self._note_pending(kind, raised=False)
            # task-31384: a kind may RETAIN its payload past teardown (the
            # approvals bridge keeps a definitive-after-start batch mounted
            # in its "finishing" phase). The hook runs OUTSIDE the lock and
            # returns True to keep the payload parked; it may take the lock
            # itself to mutate the retained payload.
            if on_teardown is None or not on_teardown():
                self.unpark_round_payload(kind, round_id)
            if session_id is not None:
                discard = getattr(self._seams, "discard_pending_round", None)
                if discard is not None:
                    discard(session_id, round_id)
            try:
                if (
                    retained_decision
                    and getattr(self._seams, "set_pending_decision", None) is not None
                ):
                    self._seams._marshal_pending_decision_projection()
                else:
                    self.remount_head(
                        kind, owning_session_id if session_id is not None else None
                    )
            except Exception:  # noqa: BLE001 -- teardown must never raise
                logger.opt(exception=True).debug(
                    f"Failed to marshal {kind} remount during teardown"
                )
