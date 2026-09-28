"""RemoteBindingStatusCache + classify_probe_result (Phase 2d, Task 13).

Pins the spec's Binding-model status rules against the pure in-memory
cache:

* **Optimistic cold start** — an unseen binding is READY with no reason,
  no identity, no timestamp; admission (Task 17/18) treats
  READY-without-identity as "admit; first ping captures identity".
* **Transport-failure-only learning** — the taxonomy kinds that mean the
  worker never accepted the root (UNREACHABLE, INTERPRETER_MISSING,
  PYTHON_TOO_OLD, WORKER_FAILED_TO_START, STDOUT_NOISE) flip the binding
  to BLOCKED; the admitted-marker kinds (OP_TIMEOUT, REMOTE_OP_FAILED)
  and MUX_ERROR leave the cached state untouched.
* **Pin failure is not BLOCKED** — it is STALE_IDENTITY, keeping the old
  identity chain for diagnosis (spec: "Identity freshness").
* **Recovery** — any success flips the cache back to READY, and a ping
  result re-captures the identity chain (the recreated-root parity
  case).
* **Probe debounce** — one scheduled probe per resolved host per 30s
  (injectable), claimed atomically so a concurrent composition pass can
  never double-schedule.
* **classify_probe_result** — the probe-result mapping Task 14 consumes:
  success frame → ready, framed ``root_pin_failed`` refusal → pin_failed
  (the worker does NOT distinguish a missing root from a rejected one —
  both emit ``root_pin_failed``, verified against the committed bundle),
  blocking transport kinds → blocked, and every state-preserving outcome
  (OP_TIMEOUT / REMOTE_OP_FAILED / MUX_ERROR / unparseable or
  non-terminal frames) → op_failed.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from tldw_chatbook.Tools.remote_binding_status import (
    BindingState,
    CachedStatus,
    RemoteBindingStatusCache,
    classify_probe_result,
)
from tldw_chatbook.Tools.remote_workspace_transport import (
    RemoteCallResult,
    TransportFailure,
    TransportFailureKind,
)

BLOCKING_KINDS = [
    TransportFailureKind.UNREACHABLE,
    TransportFailureKind.INTERPRETER_MISSING,
    TransportFailureKind.PYTHON_TOO_OLD,
    TransportFailureKind.WORKER_FAILED_TO_START,
    TransportFailureKind.STDOUT_NOISE,
]

NO_FLIP_KINDS = [
    TransportFailureKind.OP_TIMEOUT,
    TransportFailureKind.REMOTE_OP_FAILED,
    TransportFailureKind.MUX_ERROR,
]

VALID_STATES = {
    "READY", "BLOCKED", "MISSING", "STALE_IDENTITY",
}

CHAIN_A = [["/srv/work", 1, 2, 16877], ["/srv", 3, 4, 16877], ["/", 5, 6, 16877]]
CHAIN_B = [["/srv/work", 99, 98, 16877], ["/srv", 3, 4, 16877], ["/", 5, 6, 16877]]


# -- optimistic cold start ----------------------------------------------------


def test_unseen_binding_is_optimistically_ready() -> None:
    cache = RemoteBindingStatusCache()
    status = cache.status("binding-1")
    assert isinstance(status, CachedStatus)
    assert status.state == "READY"
    assert status.state == BindingState.READY
    assert status.reason is None
    assert status.identity_chain is None
    assert status.updated_monotonic is None
    # Reads of an unseen binding never materialize learning state.
    assert cache.identity_for("binding-1") is None
    assert cache.transient_failure_count("binding-1") == 0


def test_optimistic_default_survives_transient_failures() -> None:
    cache = RemoteBindingStatusCache()
    cache.record_transport_failure(
        "binding-1", TransportFailureKind.OP_TIMEOUT, "slow op", now=1.0
    )
    status = cache.status("binding-1")
    assert status.state == "READY"
    assert status.reason is None
    assert status.updated_monotonic is None  # nothing was learned
    assert cache.transient_failure_count("binding-1") == 1


# -- transport failure learning -----------------------------------------------


@pytest.mark.parametrize("kind", BLOCKING_KINDS)
def test_blocking_transport_failures_flip_to_blocked(kind) -> None:
    cache = RemoteBindingStatusCache()
    cache.record_transport_failure("binding-1", kind, "the reason", now=10.0)
    status = cache.status("binding-1")
    assert status.state == "BLOCKED"
    assert status.reason == "the reason"
    assert status.updated_monotonic == 10.0


@pytest.mark.parametrize("kind", NO_FLIP_KINDS)
def test_no_flip_kinds_leave_a_blocked_state_untouched(kind) -> None:
    cache = RemoteBindingStatusCache()
    cache.record_transport_failure(
        "binding-1", TransportFailureKind.UNREACHABLE, "host down", now=10.0
    )
    cache.record_transport_failure("binding-1", kind, "transient", now=20.0)
    status = cache.status("binding-1")
    assert status.state == "BLOCKED"
    assert status.reason == "host down"
    assert status.updated_monotonic == 10.0
    assert cache.transient_failure_count("binding-1") == 1


@pytest.mark.parametrize("kind", NO_FLIP_KINDS)
def test_no_flip_kinds_leave_a_captured_identity_untouched(kind) -> None:
    cache = RemoteBindingStatusCache()
    cache.record_success("binding-1", identity_chain=CHAIN_A, now=1.0)
    cache.record_transport_failure("binding-1", kind, "transient", now=2.0)
    assert cache.status("binding-1").state == "READY"
    assert cache.identity_for("binding-1") == CHAIN_A


def test_record_transport_failure_rejects_non_taxonomy_kind() -> None:
    cache = RemoteBindingStatusCache()
    with pytest.raises(TypeError):
        cache.record_transport_failure("binding-1", "unreachable", "not an enum")  # type: ignore[arg-type]


# -- pin failure → STALE_IDENTITY ---------------------------------------------


def test_pin_failure_is_stale_identity_not_blocked() -> None:
    cache = RemoteBindingStatusCache()
    cache.record_success("binding-1", identity_chain=CHAIN_A, now=1.0)
    cache.record_pin_failure("binding-1", now=2.0)
    status = cache.status("binding-1")
    assert status.state == "STALE_IDENTITY"
    assert status.state != "BLOCKED"
    assert status.updated_monotonic == 2.0
    # The old chain stays for diagnosis: the host was reachable.
    assert cache.identity_for("binding-1") == CHAIN_A


def test_pin_failure_on_cold_binding_is_stale_identity_without_chain() -> None:
    cache = RemoteBindingStatusCache()
    cache.record_pin_failure("binding-1", now=1.0)
    status = cache.status("binding-1")
    assert status.state == "STALE_IDENTITY"
    assert status.identity_chain is None


# -- success: recovery and identity re-capture --------------------------------


def test_success_recovers_blocked_binding() -> None:
    cache = RemoteBindingStatusCache()
    cache.record_transport_failure(
        "binding-1", TransportFailureKind.UNREACHABLE, "host down", now=1.0
    )
    cache.record_success("binding-1", now=2.0)
    status = cache.status("binding-1")
    assert status.state == "READY"
    assert status.reason is None
    assert status.updated_monotonic == 2.0


def test_success_after_stale_recaptures_identity_chain() -> None:
    cache = RemoteBindingStatusCache()
    cache.record_success("binding-1", identity_chain=CHAIN_A, now=1.0)
    cache.record_pin_failure("binding-1", now=2.0)
    cache.record_success("binding-1", identity_chain=CHAIN_B, now=3.0)
    status = cache.status("binding-1")
    assert status.state == "READY"
    assert status.identity_chain == CHAIN_B
    assert cache.identity_for("binding-1") == CHAIN_B


def test_success_without_chain_keeps_previous_capture() -> None:
    cache = RemoteBindingStatusCache()
    cache.record_success("binding-1", identity_chain=CHAIN_A, now=1.0)
    cache.record_success("binding-1", now=2.0)
    assert cache.identity_for("binding-1") == CHAIN_A


def test_missing_binding_is_recovered_by_success() -> None:
    cache = RemoteBindingStatusCache()
    cache.record_missing("binding-1", "root path absent on host", now=1.0)
    status = cache.status("binding-1")
    assert status.state == "MISSING"
    assert status.reason == "root path absent on host"
    cache.record_success("binding-1", identity_chain=CHAIN_A, now=2.0)
    assert cache.status("binding-1").state == "READY"


# -- identity storage hygiene ---------------------------------------------------


def test_identity_for_is_none_cold_and_captured_after() -> None:
    cache = RemoteBindingStatusCache()
    assert cache.identity_for("binding-1") is None
    cache.record_success("binding-1", identity_chain=CHAIN_A, now=1.0)
    assert cache.identity_for("binding-1") == CHAIN_A


def test_stored_chain_is_isolated_from_caller_mutation() -> None:
    cache = RemoteBindingStatusCache()
    chain = [["/srv/work", 1, 2, 16877]]
    cache.record_success("binding-1", identity_chain=chain, now=1.0)
    chain[0][1] = 999  # caller mutates its own copy afterwards
    assert cache.identity_for("binding-1")[0][1] == 1
    returned = cache.identity_for("binding-1")
    assert isinstance(returned, list)
    returned[0][1] = 777  # caller mutates the returned copy
    assert cache.identity_for("binding-1")[0][1] == 1
    assert cache.status("binding-1").identity_chain == [["/srv/work", 1, 2, 16877]]


# -- recovery-probe debounce ----------------------------------------------------


def test_probe_debounce_allows_one_per_window() -> None:
    cache = RemoteBindingStatusCache(probe_debounce_s=30.0)
    assert cache.should_schedule_probe("host-a", now=100.0) is True
    assert cache.should_schedule_probe("host-a", now=110.0) is False
    assert cache.should_schedule_probe("host-a", now=129.999) is False
    # The window elapsed exactly: a new probe may be scheduled.
    assert cache.should_schedule_probe("host-a", now=130.0) is True
    assert cache.should_schedule_probe("host-a", now=140.0) is False


def test_probe_debounce_interval_is_injectable() -> None:
    cache = RemoteBindingStatusCache(probe_debounce_s=5.0)
    assert cache.should_schedule_probe("host-a", now=10.0) is True
    assert cache.should_schedule_probe("host-a", now=14.9) is False
    assert cache.should_schedule_probe("host-a", now=15.0) is True


def test_probe_debounce_windows_are_per_host_key() -> None:
    cache = RemoteBindingStatusCache()
    assert cache.should_schedule_probe("host-a", now=100.0) is True
    assert cache.should_schedule_probe("host-b", now=100.0) is True
    assert cache.should_schedule_probe("host-a", now=101.0) is False
    assert cache.should_schedule_probe("host-b", now=101.0) is False
    assert cache.should_schedule_probe("host-c", now=101.0) is True


def test_mark_probe_dispatched_reanchors_the_window() -> None:
    cache = RemoteBindingStatusCache(probe_debounce_s=30.0)
    assert cache.should_schedule_probe("host-a", now=100.0) is True
    cache.mark_probe_dispatched("host-a", now=105.0)
    assert cache.should_schedule_probe("host-a", now=110.0) is False
    assert cache.should_schedule_probe("host-a", now=135.0) is True


def test_probe_claim_is_atomic_under_contention() -> None:
    cache = RemoteBindingStatusCache()
    barrier = threading.Barrier(8)
    verdicts: list[bool] = []
    lock = threading.Lock()

    def claim() -> None:
        barrier.wait()
        verdict = cache.should_schedule_probe("host-a", now=100.0)
        with lock:
            verdicts.append(verdict)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert verdicts.count(True) == 1
    assert verdicts.count(False) == 7


def test_probe_debounce_requires_positive_interval() -> None:
    with pytest.raises(ValueError):
        RemoteBindingStatusCache(probe_debounce_s=0.0)


# -- thread-safety smoke ---------------------------------------------------------


def test_thread_safety_smoke_mixed_records() -> None:
    cache = RemoteBindingStatusCache(probe_debounce_s=30.0)
    chains = (CHAIN_A, CHAIN_B)
    errors: list[Exception] = []
    error_lock = threading.Lock()

    def worker(thread_index: int) -> None:
        try:
            for step in range(100):
                cache.record_transport_failure(
                    "shared",
                    TransportFailureKind.UNREACHABLE,
                    "down",
                    now=float(step),
                )
                cache.record_transport_failure(
                    "shared",
                    TransportFailureKind.OP_TIMEOUT,
                    "slow",
                    now=float(step),
                )
                cache.record_success(
                    "shared", identity_chain=chains[thread_index % 2], now=float(step)
                )
                cache.record_pin_failure("shared", now=float(step))
                cache.record_missing("shared", "absent", now=float(step))
                status = cache.status("shared")
                assert status.state in VALID_STATES
                cache.identity_for("shared")
                cache.transient_failure_count("shared")
                cache.should_schedule_probe("host", now=100.0 + step)
        except Exception as exc:  # noqa: BLE001 - recorded and re-asserted below
            with error_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    final = cache.status("shared")
    assert final.state in VALID_STATES
    assert final.updated_monotonic is not None
    # Every thread completed (errors == []), so record_success ran and
    # the chain is one of the two captured ones; nothing ever clears it.
    assert cache.identity_for("shared") in chains
    # Probe windows were claimed atomically: at most one True per step window.
    assert cache.should_schedule_probe("host", now=1_000.0) is True


# -- classify_probe_result -------------------------------------------------------


def _frame(outcome: str, code: str) -> bytes:
    payload: dict[str, Any] = {
        "version": 1,
        "operation_id": "op",
        "outcome": outcome,
        "code": code,
        "result": None,
        "error": None,
        "elapsed_ms": 1,
        "truncated": False,
        "cleanup_proven": True,
    }
    return json.dumps(payload).encode()


def _transport_call(kind: TransportFailureKind) -> RemoteCallResult:
    return RemoteCallResult(
        admitted=False,
        response=None,
        failure=TransportFailure(kind, 255, "reason"),
    )


def test_classify_successful_ping_frame_is_ready() -> None:
    # A successful ping emits ONE frame and NO admitted marker — the
    # classifier must read the frame's outcome, never the marker bit.
    result = RemoteCallResult(
        admitted=False, response=_frame("success", "ok"), failure=None
    )
    assert classify_probe_result(result) == "ready"


def test_classify_admitted_success_frame_is_ready() -> None:
    result = RemoteCallResult(
        admitted=True, response=_frame("success", "ok"), failure=None
    )
    assert classify_probe_result(result) == "ready"


def test_classify_root_pin_refusal_is_pin_failed() -> None:
    # The worker emits root_pin_failed for BOTH a missing root and an
    # identity-mismatch refusal; there is no distinct root-absent code,
    # so both arrive here as pin_failed (Task 14 resolves MISSING).
    result = RemoteCallResult(
        admitted=False, response=_frame("failure", "root_pin_failed"), failure=None
    )
    assert classify_probe_result(result) == "pin_failed"


@pytest.mark.parametrize(
    "code", ["invalid_request", "worker_failure", "tool_failure", "unsupported_operation"]
)
def test_classify_other_refusals_preserve_state(code: str) -> None:
    # A refusal that is not the pin-refused signal says nothing about
    # binding availability: op_failed means "leave the cache alone".
    result = RemoteCallResult(
        admitted=False, response=_frame("failure", code), failure=None
    )
    assert classify_probe_result(result) == "op_failed"


@pytest.mark.parametrize("kind", BLOCKING_KINDS)
def test_classify_blocking_transport_kinds_are_blocked(kind) -> None:
    assert classify_probe_result(_transport_call(kind)) == "blocked"


@pytest.mark.parametrize("kind", NO_FLIP_KINDS)
def test_classify_state_preserving_kinds_are_op_failed(kind) -> None:
    assert classify_probe_result(_transport_call(kind)) == "op_failed"


@pytest.mark.parametrize(
    "response",
    [
        b"not json",
        b"[1, 2, 3]",
        b'{"outcome": "admitted", "code": "root_pinned"}',
        b'"success"',
    ],
)
def test_classify_unusable_frames_preserve_state(response: bytes) -> None:
    result = RemoteCallResult(admitted=False, response=response, failure=None)
    assert classify_probe_result(result) == "op_failed"


def test_classify_missing_response_and_failure_preserves_state() -> None:
    # Contract-impossible (either a frame or a failure is delivered);
    # the classifier must still answer, conservatively.
    result = RemoteCallResult(admitted=False, response=None, failure=None)
    assert classify_probe_result(result) == "op_failed"


def test_classify_failure_outranks_a_frame() -> None:
    result = RemoteCallResult(
        admitted=False,
        response=_frame("success", "ok"),
        failure=TransportFailure(TransportFailureKind.UNREACHABLE, 255, "down"),
    )
    assert classify_probe_result(result) == "blocked"
