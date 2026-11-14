# App-scoped SQLite retirement

Task: TASK-34661. Policy and this written implementation contract were approved
by the user in this chat on 2026-10-07. Execute inline in the existing worktree.

## Goal and evidence

Retire SQLite handles that this application actually owns, after their producers
settle, without closing an existing borrower's shared handle. The actual
private-profile `TldwCli` regression currently finds five constructor handles
still queryable after app shutdown. A native Darwin census finds 21 DB-related
descriptors before test cleanup; that total is not attribution to only those
five handles or proof of a physical-executor-join boundary. See
`Docs/QA/task-34661/README.md` for retained REDs and diagnostic limitations.

## Ownership contract

- Evaluations, Library Collections, Subscriptions and Workspace are the four
  app-composed stores in scope. Retire their eligible current-thread handles
  through their existing source-owner close APIs, not raw `sqlite3.close()`.
- Chat selects a process-reusable config/Notes database object. The object does
  not become app-exclusive merely because `app.chachanotes_db` references it.
  Observe the pre-existing current-thread handle without acquiring another
  connection. Preserve that exact live borrowed handle. A handle newly opened
  by app work, or an app-opened replacement after prior native retirement, can
  be retired when eligible. Keep the reusable shared database object usable.
- Only the owning thread closes an eligible handle. Foreign-thread, custom or
  uncertain ownership is not permission to close. Existing memory/custom owner
  compatibility stays outside the file-backed retirement claim.
- Preserve active transactions, managed operations and tracked live cursor
  borrowers. Existing registry/admission evidence remains authoritative; a
  handle reference or logical Worker cancellation alone is not proof of drain.

## Ordering and failure behavior

Use the existing application composition and shutdown boundaries; add no
generic lifecycle container, global owner sweep or second connection registry.
Fence and settle existing producer owners before final database retirement.
Drain Evaluations with its existing `aclose()` task boundary. Respect existing
Console, Notes, actor, capture, scheduler and worker shutdown ordering; close
stores only after the applicable shutdown work and Textual unmount have settled.

If a producer fails to settle or native eligibility cannot be proven, retain
the affected handle and its original owner/cache/lease references. If native
close fails, do not clear references or report native retirement. Preserve the
primary shutdown exception; attach metadata-only cleanup failure information
when needed. Do not discard a failed owner merely to make a second shutdown look
clean. Explicit retirement is repeat-safe after successful close.

Physical worker completion is separate from cancelled asyncio wrappers. This
change does not globally join or replace a caller's default executor; uncertain
live work stays visible rather than being declared retired.

## Verification

The existing actual-app RED is the cold-owner regression. Add focused real
SQLite controls for a warm borrowed Chat handle, an app-opened replacement,
active managed/raw transaction and tracked-cursor refusal, native-close failure,
producer failure, and repeated shutdown. Assert native behavior (successful SQL
versus a closed-native error), not only cleared Python fields or mock calls.

Run repeated actual app startup/shutdown in the existing selected private-profile
process. Observe native handles and native descriptor types at named boundaries,
including after actual executor join **before test-owned cleanup**. Cleanup after
measurement must not be credited to production. Report remaining process-owned
handles separately; no aggregate qualification claim from a quiet-body pass.

Run targeted lifecycle, existing operation-owned retirement, producer-settlement
and affected native ownership controls plus static/Backlog/artifact guards.
Retain setup-invalid attempts, diagnostic warnings and failed outcomes. New raw
aggregate logs remain local; publish only audited outcomes, hashes and non-secret
probe source. Do not run a full sweep without opt-in.

## Scope and ADR

ADR required: yes
ADR path: `backlog/decisions/225-app-scoped-sqlite-retirement.md`
Reason: explicit app-versus-process data ownership and shutdown ordering.
ADR036, ADR094 and ADR126 remain governing inputs; no schema change is needed.

The already tested cursor and mounted Interrupt test repairs remain separate;
their production transaction and cancellation contracts do not change here.
No GC/cache policy, FD limit, test deadline, profile/getter/admission or recovery
change; no dependency; no warning suppression, global quiescence, foreign-thread
close, forced executor cancellation, other-session controls, push or merge.
TASK31966/TASK31245 qualification requirements remain unwaived.

## Approved AgentRuns extension — 2026-10-07

The user approved extending this contract to the surviving app-created AgentRuns
handles. This supersedes the earlier five-store scope limit, not its retained
failures. Trace actual Console receipt/worker ownership, prove a causal native
RED, and reuse existing runtime/operation-owned close boundaries. Preserve warm
borrowers, same-thread closure, active operations and failed-close evidence.
No global sweep, second registry, foreign close or shutdown-budget expansion.
Verify actual native handles and descriptors after physical join before cleanup;
an observed creation stack alone does not attribute the surviving handle.

The implementation keeps AgentRuns at its existing Console disposal boundary.
The disposed runtime remains attached as the existing permanent intake fence;
still-live view count reads use the existing finite operation-owned guard, so
they cannot leave newly reopened caches behind after that boundary. Eligible
native retirement is checked through the existing settled-source helper. Busy
or failed retirement retains its owner/cache and reports an incomplete category
after sibling cleanup, without replacing cancellation or the primary error.

Actual reply and preview work also use the existing finite ownership helper:
reply cleanup encloses the complete physical method, not only its first DB read.
Final runtime disposal includes the controller's already tracked physical-worker
tasks in its existing bounded wait and does not cancel their owner task when the
unchanged budget expires. If native work remains, preserve its owner and report
incomplete rather than extending the budget or closing from a foreign thread.
Exact file-backed owners alone receive the stricter close-error result; custom
and in-memory compatibility stays unchanged.
