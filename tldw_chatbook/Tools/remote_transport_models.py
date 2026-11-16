"""SSH result types shared by transport and in-memory status readers."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class TransportFailureKind(enum.Enum):
    """The transport failure taxonomy (spec: "Failure taxonomy").

    The worker's admitted marker — not exit-code guessing — is the
    primary classifier: kinds above :attr:`OP_TIMEOUT` mean the worker
    never accepted the root (transport/setup class, BLOCKED-eligible for
    the status cache); :attr:`OP_TIMEOUT`/:attr:`REMOTE_OP_FAILED` mean
    the marker arrived and the operation itself failed (typed tool
    error, status unchanged). Task 13/14 consume these names verbatim.
    """

    #: ssh exit 255 with no remote output, or a marker-arrival deadline
    #: kill: unreachable, auth failure, or a stalled handshake.
    UNREACHABLE = "unreachable"
    #: Remote exit 127: the configured interpreter is not on the host.
    INTERPRETER_MISSING = "interpreter_missing"
    #: Remote exit 76: the bootstrap's version gate rejected the python.
    PYTHON_TOO_OLD = "python_too_old"
    #: Any other no-marker exit: the worker never reached framing.
    WORKER_FAILED_TO_START = "worker_failed_to_start"
    #: Admitted, then watchdog exit 75, the watchdog stderr marker, the
    #: completion-deadline kill, or any post-admission death after the
    #: op ran its full budget (the worker's own signal-killing tiers).
    OP_TIMEOUT = "op_timeout"
    #: Admitted, then any other death (signal, lost connection, EOF
    #: without a final frame).
    REMOTE_OP_FAILED = "remote_op_failed"
    #: >4KB of magicless stdout: the channel is unusable.
    STDOUT_NOISE = "stdout_noise"
    #: Mux connection/protocol failure; triggers a master restart for
    #: the NEXT call, never a retry of this one.
    MUX_ERROR = "mux_error"
    #: ``ssh -G`` no longer resolves the binding's locator to the
    #: destination recorded (and consented to) when it was added -- an
    #: ssh_config alias was retargeted. Checked before every identity
    #: capture so a pin failure can never re-capture a different host.
    DESTINATION_CHANGED = "destination_changed"


@dataclass(frozen=True)
class TransportFailure:
    """One typed transport-layer failure.

    Attributes:
        kind: The taxonomy row (see :class:`TransportFailureKind`).
        exit_code: The reaped ssh process's returncode — negative for
            signal deaths (e.g. ``-9`` after the deadline group kill),
            ``None`` only when ssh could not be spawned at all.
        reason: Static, human-readable reason (Task 13 surfaces it);
            may quote one bounded stderr line when a marker line exists
            (python version, watchdog, mux error).
    """

    kind: TransportFailureKind
    exit_code: int | None
    reason: str


@dataclass(frozen=True)
class RemoteCallResult:
    """The outcome of one ``RemoteWorkspaceTransport.call``.

    Attributes:
        admitted: Whether the admitted marker (``root_pinned``) was
            seen — the status cache's transport-vs-operation bit.
        response: The FINAL response frame bytes (magic stripped,
            newline stripped), or ``None`` on transport failure. A
            single worker-refusal frame is a response, not a transport
            failure: the worker framed its own refusal.
        failure: ``None`` when a frame was delivered; otherwise the
            typed transport failure.
    """

    admitted: bool
    response: bytes | None
    failure: TransportFailure | None
