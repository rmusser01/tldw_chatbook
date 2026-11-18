# Finite History reads retire their own native worker handle

Only the authorized replacement checkout. Raw diagnostics remain LOCAL under
`/private/tmp/console-latency-continuation-fT3VrM`; safe outcomes and
[38 verified hashes](history-worker-sha256.txt) only. No other session, shared
environment, collector, timing-limit or integration change.

## Preceding tint qualification remains NON-GREEN

d64e5c90574b86c83051d9e7b35b64dfef3acd40: fresh scale integrity/cleanup pass;
Keyword300 PASS111.760583ms P95/13.735958ms maximum loop. Original UI70 fails
52x20 preparation154.416ms, activation1/3/4 59.326/52.060/54.812ms and120x50
activation2/3/4 81.279/53.980/62.073ms, unchanged50ms limit. All original
outcomes/busy100ms/source/corpus fences retained. Post-Runner/default-executor
join Chat0/noDB; five regular log/lease/lock files, not aggregateFD0. Tint reuse
is not latency clearance or unique cause of this run's unrecorded pauses.

## Native History finding and existing ADR225 correction

The previous broad run's first real History read left four history.sqlite/
WAL/SHM descriptors, visible in85 later censuses. Isolated original-render
control has zero modal renders, four service reads, registered1->2 then2->2;
body1PASS but required descriptor gate process1. This is not a backdrop leak
or a waived baseline. The existing controller uses unscoped asyncio.to_thread
for this file-backed synchronous read, unlike its adjacent finite scope reads.

Under accepted [ADR225](../../../backlog/decisions/225-app-scoped-sqlite-retirement.md),
replace only that one dispatch with existing run_owned_db_call(db, method,
**kwargs). No new helper/registry/close policy. Original native query, async and
memory branches, custom/awaitable results, error degradation and cancellation
remain. Newly opened handles close in their original worker after the callback;
existing warm handles and active transactions remain borrowed. A cancelled
awaiter never closes an executing native read.

Initial six-control run has four genuine cold/cancellation failures and two
setup-invalid warm identity assertions: the transaction context returns a proxy,
not the native connection. Capture the actual existing native handle instead;
corrected RED4FAIL/2warmPASS7.87s, before production edit. This changes no native
operation/assertion requirement; initial evidence remains. Cold success/error
retain registered2 instead1; cancelled reads never report owning-thread closure.
Test cleanup follows the failed assertions on the same worker, not foreign close.

After the one-line change: native six plus every original Active/History case
48PASS15.55s; final native six plus exact reuse, saved-launch and manual-recovery
callers24PASS69.95s. No pytest warnings. All48 and all24 required native descriptor
censuses contain no private SQLite files. Six native cases repeat between runs;
these are72 executions, not72 unique tests. They verify literal results/errors,
closed native handles before cleanup, exact warm transaction identity, owner
thread closure, and live native SQL while the cancelled callback is held.

New test Ruff/format clean after one lambda-to-dict lint correction. Workspace
format clean;69 inherited Ruff findings unchanged/no additions, not whole-file
lint clean. AST audit reverses only the exact dispatch replacement and matches
all prior executable code. All eleven artifact guards pass. Inline author review,
not independent. Final affected Backlog/whitespace checks follow reconciliation.

Local commit/fresh exact-source qualification follows. Latency and broader native/
Windows/ordinary-quit/unfamiliar-participant/older aggregate-FD gates remain open.
TASK34661/TASK31966/TASK31245 stay In Progress; no resource or parent closeout
is manufactured from these bounded native passes. No full-suite sweep or push.
