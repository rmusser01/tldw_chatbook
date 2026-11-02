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

Abandoned warm-up calls the retained transcriber's existing `cleanup()` only
after its native call returns, then releases capture resources. No joining or
cleanup is attempted by Cancel while that call is live. Failed cleanup retains
the exact work record and closes admission until exit; retries say cleanup
failed and require quitting/restarting. Maintenance does not claim readiness.
The shared active record and the session's record are the same object, not
separate registries. Successful handoff releases preparation admission before
normal capture lifecycle takes over.

Only positively attributed plain-HTTP artifacts earn byte progress and an
inactivity deadline. Unknown transport, Xet, provider cache and native loading
remain opaque Loading with responsive Cancel. A repository-wide or cache-wide
sum is not attempt attribution. Remove this PR's process-wide Xet override;
respect the user's environment. Tqdm's existing lock repair remains independent.

Rejected: releasing admission on logical Cancel (overlapping native loads),
joining teardown (unresponsive quit), guessing stalls from quiet caches (false
failures), and a second registry of screen sessions (duplicated lifetime owners).
The broad automatic-bound AC and unsupported native qualification stay open.
