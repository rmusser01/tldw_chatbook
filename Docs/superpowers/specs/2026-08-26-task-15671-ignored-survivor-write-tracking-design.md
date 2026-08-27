# TASK-15671: Ignored survivor-write tracking — Design

Date: 2026-08-26
Status: Ready for user review
Task: TASK-15671
Governed by: ADR-089, ADR-092

## Problem

Console change review respects `.gitignore` except for paths named by recorded
WRITE-tool steps: those paths are force-added at a turn boundary so an agent
cannot hide a direct edit by choosing an ignored path.

The post-turn survivor window closes without those paths. A child can therefore
create an ignored file after its parent returns and leave no review card even
though its WRITE step is durable. Reading every child step only at close is not
enough: tool-call steps precede execution, children can cross several turn
boundaries, and a successor baseline is immutable once taken.

## Constraints

- Parent E, survivor E / successor B, and successor E remain exact abutting Git
  SHAs under ADR-089.
- Only attributed sub-agent WRITE `tool_call` paths qualify. Read tools and
  unrecorded script side effects remain excluded.
- A child can be pending before its thread enters the model scope and can
  finish while a boundary snapshot is in flight. Both states must remain
  visible to the continuation lifecycle.
- A child may outlive multiple parent turns. Its path state follows that child;
  a later turn's newly spawned children cannot leak backward.
- Existing primary-turn path eligibility, within-root checks, size-cap refusal,
  and review output remain unchanged.
- Successful synchronous `write_file` calls are in scope. Detached filesystem
  work that continues after both a tool timeout and the child lifecycle ends is
  not a completed in-lifecycle WRITE and remains outside this guarantee.
- Change tracking is best-effort and never breaks a reply or child teardown.

## Decision

### 1. Keep one small child-change state per spawning turn

Each `run_reply` creates a private `_ChildChangeState` shared by that turn's
`on_step`, child scope, and settle callback. It contains:

- an opaque owner key for the spawning turn;
- attributed child run IDs;
- de-duplicated normalized WRITE paths; and
- the owner's current child-scope count.

The attributed callback projects only
`ChangeTurnTracker.tool_touched_paths((step,))`. Relative tool arguments are
resolved against the captured scratch root, matching the real file tool;
absolute arguments remain absolute. Production Console runs always have that
private scratch root. Test/non-production callers that omit it retain the
existing raw-relative tracker semantics rather than inventing a second lookup
of the configurable tool fallback. The tracker still performs canonical
within-root validation. AgentService persists the step before invoking this
callback, so no step payload or file content is duplicated.

The state object, rather than a step watermark, bridges time. A WRITE intent
observed before E remains available after E. If its file does not exist yet,
the next boundary retries the force-add. If it exists, Git records it at the
earliest boundary that sees it. Replaying a path is harmless after it becomes
tracked.

### 2. Make pending children visible before parent E returns

The child thread can be scheduled after `thread.start()` returns, so scope entry
is not the registration boundary. Before parent E, the bridge queries the
current `AgentService.live_subagent_handles()`. If any handle is still pending
or running, it registers the current turn's state in a conversation-keyed live
state map before taking E. Thus a delayed child is visible to the parent E,
the opened survivor window, and any successor B before its scope begins.

The state is also registered on ordinary child-scope entry. The turn's settle
callback is bound to both the state and its service; after a child settles, it
removes the state from the live map only when that service reports no remaining
live handles. A survivor window retains its own references, so map removal
cannot erase already-owned paths.

No AgentService launch contract changes. The existing handle registry supplies
the parent-thread happens-before seam and naturally excludes launch failures,
which have no live handle by E.

### 3. Capture state references before E, then open after E

At turn start, the bridge snapshots references to already-live states. These
are inherited survivors, and their current paths are passed to successor B.

Immediately before turn E, the bridge captures the states that are pending or
live at that instant and copies their current paths. Turn E receives the union
of existing primary WRITE paths, inherited states retained from B, and the
current turn's child state. The current direct-turn force-add rules are reused
unchanged.

If any child state was pending/live in that pre-E capture, the bridge opens the
post-turn window after E with those exact state references even if every child
finished during the snapshot. It then performs the existing immediate
liveness recheck and closes the window if none remain. A path published while E
was blocked is therefore retried by that immediate continuation close instead
of falling through the boundary.

A new post-turn window holds all states live at its E, including inherited
ones. Later callbacks mutate those same objects. A child spawned by a later
turn has a different state and cannot contribute to the older window.

### 4. Atomically associate a successor before B starts

Starting B and only afterward attaching its handle leaves a race in which a
child can choose a fresh survivor E first. The bridge replaces that ordering
with a small `_SuccessorBoundaryClaim`:

1. under `_change_window_lock`, successor startup either installs a pending
   claim on the open, non-closing survivor window and copies that window's
   retained state references, or observes that fresh close is already in
   progress;
2. an installed claim exists before `begin_turn` can start B; the returned
   handle (or failure) is attached to the claim and its event is released;
3. a survivor closer that sees a claim waits outside the bridge lock for its
   handle, then uses that exact B; and
4. successor B force-adds paths from both the live-state snapshot and the
   claimed window references, so final-settle removal cannot move a pre-B write
   into successor E; and
5. if fresh close already owns the window, successor startup waits for its
   completion before starting B, so B naturally uses the freshly advanced tip.

Waits use the existing bounded snapshot timeout. A timeout disables tracking
for the successor turn and logs the tracking failure; it never falls back to an
overlapping boundary. Claim and close events are released in `finally`.

### 5. Force recorded paths at B and preserve late paths for successor E

`ChangeTurnTracker.begin_turn` gains optional `touched_paths=()`. For each root
it applies the existing within-root and size-cap filters, then asks the shadow
repository for one snapshot that force-adds eligible existing paths inside the
snapshot's existing non-reentrant repository lock. `ShadowRepo.snapshot` gains
an optional force-path argument and performs `add -f` inline; it must not call
the public `force_add` while holding the same lock.

Existing callers omit the argument and behave identically. Fresh-E
`end_turn` uses the same atomic primitive while retaining existing output.

When a survivor closes against supplied successor SHAs, those commits remain
immutable. The supplied-SHA branch applies the same eligible force-add to the
shadow index without replacing B. A path first available after B is therefore
captured by successor E and shown with the existing concurrent-subagent
disclosure.

`_PostTurnChangeWindow` also owns a close-completion event and closing flag.
The first closer performs DB/Git work; later closers wait outside the bridge
lock. Consequently successor E cannot overtake close-time index priming.

## Failure behavior

- Path projection remains inside the best-effort step callback.
- Tracker/root failures keep the existing tracking disclosure.
- Successor-claim and close-completion events release on every path.
- A boundary wait timeout disables that turn's tracking rather than inventing
  overlapping history.
- No schema, durable path cache, watcher, or background poller is added.

## Testing

The production-shaped regression uses a new, non-hidden ignored file named
`ignored-agent-output.txt`; hidden dotted paths such as `.env` are correctly
rejected by the real file tool. It will:

- ignore but not pre-create the target;
- pass its absolute path to the real tool;
- explicitly enable and approve `write_file` and bind the read-write
  workspace/scratch root;
- gate a child so execution starts after parent `run_reply` returns;
- require a successful tool-result step;
- assert sentinel content on disk and at the recorded snapshot SHA; and
- assert the parent's `subagent_post_turn` snapshot lists the file.

Focused tests additionally prove:

- existing ordinary-turn ignored-path and oversize behavior stays green;
- a child stalled before scope entry is retained by its pending fleet handle
  and its later ignored write is reviewed;
- a child publishes a path and exits while E is deliberately blocked, after
  which the immediate survivor close captures it;
- successor startup claims the old window before B, while a B-start race with
  an already-running fresh close waits and preserves exact abutting SHAs;
- a barrier between final-state-map removal and successor claim proves B still
  consumes the claimed window's retained paths;
- a path known before B lands in the survivor diff and survivor E equals
  successor B;
- a new ignored file after B is primed by supplied-SHA closure and appears at
  successor E even with concurrent close callers;
- an inherited state remains visible through successor E and a second survivor
  window while that successor's own child cannot leak backward; and
- injected claim-setup and close-time tracker failures release waiting callers
  promptly without breaking reply or child teardown; and
- removing the survivor carve-out makes the real-tool regression fail.

Implementation removes the obsolete named-gap paragraph from
`_close_post_turn_change_window` outright, satisfying AC4.

Run focused tracker, survivor-boundary, child-scope-ordering, and new regression
tests. A full-suite sweep requires explicit permission and is not planned for
this isolated fix.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/092-console-live-child-write-path-boundaries.md`

Reason: the fix adds an optional cross-module tracker input and defines the
shadow-index and successor-handoff semantics around supplied SHAs. ADR-092
records that contract; ADR-089 continues to own user-visible per-turn review.

## Alternatives rejected

### Read all child runs only when the window closes

Run identity alone has no execution boundary. Tool calls precede writes,
inherited survivors cross later E snapshots, and close-time hydration cannot
repair an ignored successor B.

### Freeze per-run step indexes at parent E

An index marks WRITE intent, not completion. A small live path set lets every
adjacent boundary retry and leaves actual ownership to Git.

### Register only when the child thread enters its scope

Thread scheduling can delay scope entry until after parent return. The already
existing live-handle registry is the parent-visible source for pending work.

### Rewrite successor B after closure

Replacing the shared SHA violates ADR-089 and risks double attribution. A
pre-B claim plus index priming preserves immutable history.

### Add a filesystem watcher

That would turn bounded turn review into open-ended monitoring for detached
side effects. This task covers successful recorded WRITEs within child life.
