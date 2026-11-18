# ADR-284: Console model preparation retains physical ownership

Date: 2026-11-01
Status: Accepted
Scope: Console speech-model preparation (TASK-33621.21/.110/.112).

Cancel, observed-download timeout and screen teardown retire a logical attempt;
they cannot stop a blocking provider/native load. Retain its exact service,
daemon thread, completion event and error until physical completion. A single
process-local admission lock serializes Console model preparation, including
across replacement controllers/screens. Release admission only after the call
and any deferred service disposal finish. Never join this daemon on UI exit.

The existing screen controller's `_maintenance_sessions` retains retired
sessions, but is itself screen-owned. It cannot fence a replacement screen.
Keep that maintenance owner and extend session readiness to observe its own
warm-up record; use one shared admission lock, not another session registry.
Process scope is deliberately conservative: multiple apps in one interpreter
also serialize. Per-model concurrency requires proof of independent resources.

Abandoned work cannot open a microphone, publish success or clear another work
record. A retry during retained preparation receives a plain pending-work
message. After completion a fresh service may reuse the provider's disk cache;
this is not a guarantee of native provider resume support. Hung opaque work
may hold admission until process exit. A killable provider boundary would be
required for a stronger cancellation/automatic-bound contract.

Abandoned warm-up releases capture resources after its native call returns,
then calls the retained transcriber's existing `cleanup()` once users retire. No joining or
cleanup is attempted by Cancel while that call is live. Failed cleanup retains
the exact work record and closes admission until exit; retries say cleanup
failed and require quitting/restarting. Maintenance does not claim readiness.
The shared active record and the session's record are the same object, not
separate registries. The same daemon retains admission through capture startup
and controller publication. Warm-up return alone is not an ownership transfer.
Cancel signals retirement without waiting for either initializer; only actual
completion plus disposal, or publication of the live service to the controller,
releases admission. Retirement and that final transfer share the controller lock,
which is never held across native initialization or microphone startup.

The real lazy service accepts an optional capture-cancellation predicate. Console
installs it before preparation; other callers retain their existing behavior.
The service rechecks after blocking audio/streaming initialization and immediately
before microphone admission. A device call already admitted cannot be preempted;
the retained daemon releases that late result without publishing success.
For an already-admitted capture, that daemon first requests nonjoining service
release, then waits for the service's existing maintenance readiness (including
its actual processor threads) before disposing the transcriber. Admission stays
held throughout. A stuck processor can therefore retain it until process exit;
the daemon never joins itself or transfers that wait to the UI/executor.

An opt-out or deferred start first checks shared admission and refuses an already
retained load or failed cleanup. That synchronized check is only a snapshot:
no-owner opt-out/deferred startup does not acquire a preparation lifetime owner,
so universal serialization and automatic bounds for those paths are not claimed.
The general opt-out limitation remains TASK-33621.111.

Only positively attributed plain-HTTP artifacts earn byte progress and an
inactivity deadline. Unknown transport, Xet, provider cache and native loading
remain opaque Loading with responsive Cancel. A repository-wide or cache-wide
sum is not attempt attribution. Remove this PR's process-wide Xet override;
respect the user's environment. Tqdm's existing lock repair remains independent.

Rejected: releasing admission on logical Cancel (overlapping native loads),
joining teardown (unresponsive quit), guessing stalls from quiet caches (false
failures), and a second registry of screen sessions (duplicated lifetime owners).
The broad automatic-bound AC and unsupported native qualification stay open.
