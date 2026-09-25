"""In-memory status cache for remote workspace bindings (Phase 2d, Task 13).

Spec: ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``,
"Status semantics", "Status cache learning — transport failures only", and
"Automatic recovery". The rules this module exists to enforce:

- **Optimistic cold start.** An unseen binding reads READY with no
  reason, no identity, and no timestamp. Admission (Task 17/18) treats
  READY-without-identity as "admit; the first ping captures identity":
  a dead host costs one typed mid-run error, never a pre-flight poll.
- **Learning is transport-failure-only, classified by the taxonomy.**
  The kinds that mean the worker never accepted the root
  (UNREACHABLE, INTERPRETER_MISSING, PYTHON_TOO_OLD,
  WORKER_FAILED_TO_START, STDOUT_NOISE) flip the binding to BLOCKED.
  The operation-class kinds (OP_TIMEOUT, REMOTE_OP_FAILED) and MUX_ERROR
  leave the cached state untouched: a slow ``fs_grep`` on an otherwise
  healthy host must not take the root out of the next send, and a mux
  error's answer is the failure-triggered master restart (if the
  restart's own connect fails, THAT call's failure flips the state).
  Transient (non-flipping) failures are counted, not applied.
- **A pin failure is not BLOCKED.** It is STALE_IDENTITY — identity
  stale but host reachable — and keeps the previously captured chain for
  diagnosis (spec: "Identity freshness").
- **Recovery is the flip side of degradation.** Any success flips the
  cache back to READY, and a ping's identity chain re-captures the
  cached identity (the recreated-root parity case). MISSING records a
  probe that connected but found the root path absent.
- **Probe scheduling is debounced and atomic.**
  :meth:`RemoteBindingStatusCache.should_schedule_probe` test-and-sets
  the per-host window under the cache lock, so concurrent composition
  passes can never double-schedule the recovery probe.

Pure stdlib; no I/O; the only clocks are the injectable ``now``
parameters (defaulting to ``time.monotonic``).
"""

from __future__ import annotations

import copy
import enum
import json
import threading
import time
from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Tools.remote_workspace_transport import (
    RemoteCallResult,
    TransportFailureKind,
)

__all__ = [
    "BLOCKING_TRANSPORT_KINDS",
    "NO_FLIP_TRANSPORT_KINDS",
    "PROBE_BLOCKED",
    "PROBE_MISSING",
    "PROBE_OP_FAILED",
    "PROBE_PIN_FAILED",
    "PROBE_READY",
    "BindingState",
    "CachedStatus",
    "ProbeClassification",
    "RemoteBindingStatusCache",
    "cached_status_display",
    "classify_probe_result",
    "get_remote_binding_status_cache",
    "set_remote_binding_status_cache",
]

#: Default recovery-probe debounce: one scheduled probe per resolved
#: host per 30 seconds (spec: "Automatic recovery").
DEFAULT_PROBE_DEBOUNCE_SECONDS = 30.0

#: Transport failure kinds that mean the worker never accepted the
#: root — BLOCKED-eligible for the cache (spec: "Status cache
#: learning", the taxonomy rows above OP_TIMEOUT).
BLOCKING_TRANSPORT_KINDS = frozenset(
    {
        TransportFailureKind.UNREACHABLE,
        TransportFailureKind.INTERPRETER_MISSING,
        TransportFailureKind.PYTHON_TOO_OLD,
        TransportFailureKind.WORKER_FAILED_TO_START,
        TransportFailureKind.STDOUT_NOISE,
    }
)

#: Kinds that leave the cached state untouched: the admitted-marker
#: operation failures, and the mux error whose response is the
#: failure-triggered master restart (never a state flip).
NO_FLIP_TRANSPORT_KINDS = frozenset(
    {
        TransportFailureKind.OP_TIMEOUT,
        TransportFailureKind.REMOTE_OP_FAILED,
        TransportFailureKind.MUX_ERROR,
    }
)

#: The worker's framed refusal code for a root that would not pin or
#: capture (committed bundle: both the ping capture path and the pin
#: path emit exactly this code — the worker does NOT distinguish a
#: missing root from a rejected one).
_ROOT_PIN_FAILED_CODE = "root_pin_failed"


class BindingState(enum.StrEnum):
    """The cached state of one remote binding.

    StrEnum so members compare equal to (and format as) their plain
    string values — ``status.state == "READY"`` holds for callers that
    never import this class.
    """

    #: Optimistic default; admission allows the binding.
    READY = "READY"
    #: Transport-class failure; excluded from run roots.
    BLOCKED = "BLOCKED"
    #: Probe connected but the root path is absent on the host.
    MISSING = "MISSING"
    #: The worker refused the pinned identity; the host was reachable
    #: and the old identity chain is retained for diagnosis.
    STALE_IDENTITY = "STALE_IDENTITY"


# Probe-result classifications (Task 14 consumes these).
PROBE_READY = "ready"
PROBE_BLOCKED = "blocked"
PROBE_MISSING = "missing"
PROBE_PIN_FAILED = "pin_failed"
PROBE_OP_FAILED = "op_failed"

ProbeClassification = Literal[
    "ready", "blocked", "missing", "pin_failed", "op_failed"
]


@dataclass(frozen=True)
class CachedStatus:
    """The read-side snapshot of one binding's cached status.

    Attributes:
        state: The current :class:`BindingState`.
        reason: Human-readable reason for a non-READY state; ``None``
            when READY or cold. Transient (non-flipping) failures never
            touch this field — they only bump the internal counter
            exposed by :meth:`RemoteBindingStatusCache.transient_failure_count`.
        identity_chain: The last ping-captured identity chain (the ping
            payload's ``identity_chain``: root-first
            ``[path, device, inode, mode]`` entries), or ``None`` while
            no ping has captured one.
        updated_monotonic: ``time.monotonic()`` (or the injected
            ``now``) of the last state-APPLYING record; ``None`` cold.
    """

    state: BindingState
    reason: str | None = None
    identity_chain: list[list[object]] | None = None
    updated_monotonic: float | None = None


#: The shared optimistic cold-start snapshot: an unseen binding NEVER
#: materializes a record just because someone read it.
_COLD_STATUS = CachedStatus(
    state=BindingState.READY,
    reason=None,
    identity_chain=None,
    updated_monotonic=None,
)

#: Fixed reason recorded with STALE_IDENTITY (``record_pin_failure``
#: takes no caller reason: the refusal frame is the whole story).
_PIN_REFUSED_REASON = "worker refused the pinned root identity"


@dataclass
class _BindingRecord:
    """The mutable internal record; only ever touched under the lock."""

    state: BindingState = BindingState.READY
    reason: str | None = None
    identity_chain: list[list[object]] | None = None
    updated_monotonic: float | None = None
    transient_failures: int = 0


def _moment(now: float | None) -> float:
    """Resolve one record's timestamp: injected ``now`` or the clock."""
    return time.monotonic() if now is None else float(now)


#: Process-wide status cache shared by every composition/admission site
#: (Phase 4a). One cache per process is the point: run composition, the
#: per-call client guard, the transport's learning writes, and the
#: debounced recovery probe must all see the SAME binding states or
#: availability decisions fragment per call site. Lazily constructed;
#: ``set_remote_binding_status_cache`` is the test seam.
_APP_STATUS_CACHE: RemoteBindingStatusCache | None = None
_APP_STATUS_CACHE_LOCK = threading.Lock()


def get_remote_binding_status_cache() -> RemoteBindingStatusCache:
    """Return the process-wide :class:`RemoteBindingStatusCache`.

    Lazily constructed with the shipped defaults on first use; stable
    thereafter. Pure in-memory, no I/O, no clocks at construction --
    safe to call from any hot path (composition, note building).
    """
    global _APP_STATUS_CACHE
    if _APP_STATUS_CACHE is not None:
        return _APP_STATUS_CACHE
    with _APP_STATUS_CACHE_LOCK:
        if _APP_STATUS_CACHE is None:
            _APP_STATUS_CACHE = RemoteBindingStatusCache()
    return _APP_STATUS_CACHE


def set_remote_binding_status_cache(
    cache: RemoteBindingStatusCache | None,
) -> None:
    """Install or clear the process-wide status cache (test seam).

    Args:
        cache: The replacement cache, or ``None`` to drop the singleton
            so the next :func:`get_remote_binding_status_cache` builds a
            fresh one.
    """
    global _APP_STATUS_CACHE
    with _APP_STATUS_CACHE_LOCK:
        _APP_STATUS_CACHE = cache


def cached_status_display(
    binding_id: str, cache: RemoteBindingStatusCache | None = None
) -> str:
    """One binding's cached state as the UI live-status word (pure read).

    The Task 20 display vocabulary, one place so Settings rows, the
    Console working-folder picker, and the Alt+W switcher cannot drift:

    - ``READY`` (including the optimistic cold start) → ``"ready"``
    - ``BLOCKED`` → ``"unreachable ({reason})"`` — the transport-class
      bucket; the bounded reason string already names the specific
      failure ("unreachable or auth failed", "host lacks python3",
      "python ≥ 3.10 required (found 3.8.5)", ...)
    - ``MISSING`` → ``"missing on host"``
    - ``STALE_IDENTITY`` → ``"identity stale"``

    Never raises and never probes: an unreadable cache degrades to the
    optimistic ``"ready"``.
    """
    try:
        active = cache if cache is not None else get_remote_binding_status_cache()
        cached = active.status(binding_id)
        state = str(getattr(cached, "state", "") or BindingState.READY)
    except Exception:  # noqa: BLE001 - display-only, degrade optimistically
        return "ready"
    if state == str(BindingState.BLOCKED):
        reason = str(getattr(cached, "reason", "") or "").strip()
        return f"unreachable ({reason})" if reason else "unreachable"
    if state == str(BindingState.MISSING):
        return "missing on host"
    if state == str(BindingState.STALE_IDENTITY):
        return "identity stale"
    return "ready"


class RemoteBindingStatusCache:
    """Thread-safe, pure in-memory status cache for remote bindings.

    One ``threading.RLock`` guards every public method; records are
    internal and never handed out by reference (identity chains are
    deep-copied on the way in and out, so a caller mutating its copy
    can never corrupt the cache).

    Args:
        probe_debounce_s: Recovery-probe debounce window in seconds
            (spec default 30.0); injectable so tests can run windows
            fast. Must be positive.
    """

    def __init__(self, *, probe_debounce_s: float = DEFAULT_PROBE_DEBOUNCE_SECONDS) -> None:
        if probe_debounce_s <= 0:
            raise ValueError("probe_debounce_s must be positive")
        self._lock = threading.RLock()
        self._probe_debounce_s = float(probe_debounce_s)
        self._records: dict[str, _BindingRecord] = {}
        self._probe_dispatched: dict[str, float] = {}

    # -- reads --------------------------------------------------------------

    def status(self, binding_id: str) -> CachedStatus:
        """Return the cached status of ``binding_id`` (never materializes).

        An unseen binding returns the optimistic READY snapshot — no
        reason, no identity, no timestamp.
        """
        with self._lock:
            record = self._records.get(binding_id)
            if record is None:
                return _COLD_STATUS
            return CachedStatus(
                state=record.state,
                reason=record.reason,
                identity_chain=copy.deepcopy(record.identity_chain),
                updated_monotonic=record.updated_monotonic,
            )

    def identity_for(self, binding_id: str) -> list[list[object]] | None:
        """The last ping-captured identity chain, or ``None`` if never."""
        with self._lock:
            record = self._records.get(binding_id)
            if record is None or record.identity_chain is None:
                return None
            return copy.deepcopy(record.identity_chain)

    def transient_failure_count(self, binding_id: str) -> int:
        """How many non-flipping failures were recorded (diagnostic).

        Transient failures (OP_TIMEOUT, REMOTE_OP_FAILED, MUX_ERROR) do
        not change the cached state; this counter is the only trace.
        """
        with self._lock:
            record = self._records.get(binding_id)
            return 0 if record is None else record.transient_failures

    # -- learning -----------------------------------------------------------

    def record_transport_failure(
        self,
        binding_id: str,
        kind: TransportFailureKind,
        reason: str,
        *,
        now: float | None = None,
    ) -> None:
        """Apply one typed transport failure per the taxonomy rules.

        BLOCKING_TRANSPORT_KINDS flip the binding to BLOCKED with
        ``reason``; NO_FLIP_TRANSPORT_KINDS (and, conservatively, any
        future taxonomy member not yet classified here) leave the state,
        reason, and timestamp untouched, incrementing the transient
        counter instead.

        Args:
            binding_id: The binding the failed call ran against.
            kind: The transport failure kind (Task 11's taxonomy).
            reason: Bounded human-readable failure reason.
            now: Injectable ``time.monotonic`` for the update stamp.

        Raises:
            TypeError: If ``kind`` is not a
                :class:`~tldw_chatbook.Tools.remote_workspace_transport.TransportFailureKind`.
        """
        if not isinstance(kind, TransportFailureKind):
            raise TypeError(
                f"kind must be a TransportFailureKind, got {type(kind).__name__}"
            )
        with self._lock:
            record = self._records.setdefault(binding_id, _BindingRecord())
            if kind in BLOCKING_TRANSPORT_KINDS:
                record.state = BindingState.BLOCKED
                record.reason = reason
                record.updated_monotonic = _moment(now)
            else:
                # OP_TIMEOUT / REMOTE_OP_FAILED / MUX_ERROR (and any
                # not-yet-classified future kind): state preserved.
                record.transient_failures += 1

    def record_pin_failure(self, binding_id: str, *, now: float | None = None) -> None:
        """Mark the cached identity stale (the worker refused the pin).

        STALE_IDENTITY, not BLOCKED: the host was reachable — the old
        identity chain is kept for diagnosis, and the recovery probe's
        ping re-captures it (spec: "Identity freshness").
        """
        with self._lock:
            record = self._records.setdefault(binding_id, _BindingRecord())
            record.state = BindingState.STALE_IDENTITY
            record.reason = _PIN_REFUSED_REASON
            record.updated_monotonic = _moment(now)

    def record_success(
        self,
        binding_id: str,
        identity_chain: list[list[object]] | None = None,
        *,
        now: float | None = None,
    ) -> None:
        """Flip the binding to READY (recovery); optionally re-capture identity.

        Any success — op or probe — flips the cache back to READY. When
        ``identity_chain`` is provided (a ping result), it replaces the
        stored chain (the recreated-root parity case); ``None`` keeps
        the previously captured chain.
        """
        with self._lock:
            record = self._records.setdefault(binding_id, _BindingRecord())
            record.state = BindingState.READY
            record.reason = None
            record.updated_monotonic = _moment(now)
            if identity_chain is not None:
                record.identity_chain = copy.deepcopy(identity_chain)

    def record_missing(
        self, binding_id: str, reason: str, *, now: float | None = None
    ) -> None:
        """Mark the binding MISSING: the probe connected, the root is absent."""
        with self._lock:
            record = self._records.setdefault(binding_id, _BindingRecord())
            record.state = BindingState.MISSING
            record.reason = reason
            record.updated_monotonic = _moment(now)

    # -- recovery-probe scheduling -------------------------------------------

    def should_schedule_probe(self, host_key: str, now: float | None = None) -> bool:
        """Whether a recovery probe for ``host_key`` should run NOW.

        Test-and-set under the lock: returns ``True`` exactly once per
        debounce window per resolved host — the winning caller's True
        claims the window, so concurrent calls (the double-schedule
        race) all get ``False`` without separate bookkeeping.

        Args:
            host_key: The resolved host identity (caller-rendered key;
                per-host independence is by string).
            now: Injectable ``time.monotonic`` timestamp.
        """
        with self._lock:
            moment = _moment(now)
            last = self._probe_dispatched.get(host_key)
            if last is not None and (moment - last) < self._probe_debounce_s:
                return False
            self._probe_dispatched[host_key] = moment
            return True

    def mark_probe_dispatched(self, host_key: str, now: float | None = None) -> None:
        """Explicitly mark a probe dispatched at ``now`` (bookkeeping).

        Optional alongside :meth:`should_schedule_probe` (which claims
        the window itself): re-anchors the debounce window for callers
        that dispatch on their own schedule — e.g. a retry after a
        failed dispatch attempt.
        """
        with self._lock:
            self._probe_dispatched[host_key] = _moment(now)


def classify_probe_result(result: RemoteCallResult) -> ProbeClassification:
    """Map one probe call's :class:`~.RemoteCallResult` for the cache.

    The vocabulary Task 14 dispatches on:

    - ``"ready"`` — a terminal success frame (a successful ping emits
      ONE frame and NO admitted marker, so the frame's ``outcome`` is
      read, never the marker bit).
    - ``"pin_failed"`` — a framed refusal with code ``root_pin_failed``.
      The committed worker does NOT distinguish a missing root from a
      rejected identity (both the ping-capture path and the pin path
      emit this single code), so both arrive here; ``"missing"`` is the
      reserved classification for a worker that one day distinguishes
      them, and MISSING is otherwise recorded by Task 14's probe logic.
    - ``"blocked"`` — a BLOCKING_TRANSPORT_KINDS failure.
    - ``"op_failed"`` — every state-preserving outcome: OP_TIMEOUT (a
      probe that ran past its deadline retries next window),
      REMOTE_OP_FAILED, MUX_ERROR, and any frame the classifier cannot
      use (unparseable, non-dict, admitted-only, or a refusal code that
      is not the pin-refused signal). Conservative by construction: no
      classification, no state change.

    Args:
        result: The transport outcome of one probe call.

    Returns:
        One of ``"ready"``, ``"blocked"``, ``"missing"``,
        ``"pin_failed"``, ``"op_failed"``.
    """
    if result.failure is not None:
        if result.failure.kind in BLOCKING_TRANSPORT_KINDS:
            return PROBE_BLOCKED
        return PROBE_OP_FAILED
    if result.response is None:
        # Contract-impossible (a frame or a failure is delivered);
        # still answer conservatively.
        return PROBE_OP_FAILED
    try:
        payload = json.loads(result.response)
    except ValueError:
        return PROBE_OP_FAILED
    if not isinstance(payload, dict):
        return PROBE_OP_FAILED
    outcome = payload.get("outcome")
    if outcome == "success":
        return PROBE_READY
    if outcome == "failure" and payload.get("code") == _ROOT_PIN_FAILED_CODE:
        return PROBE_PIN_FAILED
    return PROBE_OP_FAILED
