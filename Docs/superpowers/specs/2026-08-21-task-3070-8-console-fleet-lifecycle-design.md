# TASK-3070.8 Console Fleet and Wake Lifecycle Controller Design

## Summary

Extract the approved 16-method fleet/wake family from `ChatScreen` into a new
DOM-free `ConsoleFleetLifecycleController` in
`tldw_chatbook/UI/Console_Modules/fleet.py`. The controller becomes the single
owner of completion handoff, durable unseen-marker derivation, mount-time wake
claiming, wake retry/delivery state, teardown accounting, and the survivor timer.
`ChatScreen` retains only lifecycle sequencing and mounted presentation; all
production callers address `screen._fleet` directly.

The extraction is behavior-preserving. It does not add a new user feature, storage
format, setting, dependency, or public command.

## Context and Constraints

TASK-3070.8 implements the already-approved fleet/wake boundary in
`Docs/superpowers/specs/2026-08-13-console-decomposition-wave6-design.md` and
`DESIGN.md` section 7. The immutable planning base is
`0a8e2882588fdad5a99aca6e2215735c43927528` (`origin/dev` on 2026-08-21).
The source-inspected family is 401 physical definition lines across 16 direct
`ChatScreen` methods. The existing Wave 6 architecture manifest already names
`fleet.py`, `ConsoleFleetLifecycleController`, and screen owner slot `_fleet`.

Focused baseline evidence is green: 17 tests passed across the Wave 6 compatibility
gate, survivor-tick behavior, restart staging, and wake wiring. Four existing
dependency/runtime warnings were emitted; no test failed.

The controller must:

- hold no `ChatScreen`, widget tree, sibling controller, or DOM query capability;
- receive keyword-only, narrowly named, late-bound dependencies;
- preserve synchronous mount claiming before the first view-clear/tab sync;
- preserve pending-handoff claim/release/acknowledge behavior and activation order;
- preserve user-wins-ties and displayed-screen semantics;
- preserve durable-mark-before-view-clear ordering;
- preserve teardown snapshot/leave/stage ordering;
- preserve survivor-timer arming, idempotence, cadence, and final settle paint;
- leave notification rendering and Textual lifecycle entry points on the screen.

## Architecture

### New controller

`ConsoleFleetLifecycleController` is constructed in
`tldw_chatbook/UI/Console_Modules/wiring.py` and stored at `screen._fleet`.
Its constructor is keyword-only. Dependencies are callables or service accessors,
not a screen object. The exact constructor surface is finalized from source during
planning, but it is limited to these owned edges:

- app and pending-handoff access;
- current/ensured chat store and chat controller access;
- agent-bridge readiness and fleet-wake access;
- workspace activation and session switching;
- displayed-composer resolution and displayed-screen status;
- UI-loop deferral, worker scheduling, and native-console repaint;
- interval creation and UI-timer bookkeeping;
- transcript-timer activity;
- runtime leave and app-level teardown-notice staging.

Every callback is evaluated when the controller operation runs. Wiring must not
capture a controller, store, composer, active session, or configuration value eagerly.
The controller may invoke methods on an already-resolved composer or service object,
but it never locates that object through `query`, `query_one`, `screen`, `_workspace`,
`_session`, `_agent`, or another controller field.

### Exact moved inventory

These definitions move from `ChatScreen` and do not remain as delegates or aliases:

1. `consume_pending_console_fleet_completion`
2. `_claim_console_fleet_wake_marks`
3. `_console_wake_user_priority`
4. `_console_wake_probe_composer`
5. `_console_screen_displayed`
6. `_console_wake_conversation_in_view`
7. `_poke_console_wake_retry`
8. `_on_console_wake_delivery_started`
9. `_console_wake_turn_active`
10. `_record_console_fleet_teardown`
11. `_console_fleet_unseen_ids`
12. `_console_run_marker_with_unseen`
13. `_console_fleet_survivors_live`
14. `_maybe_start_console_fleet_survivor_tick`
15. `_stop_console_fleet_survivor_tick`
16. `_console_fleet_survivor_tick`

Controller-private helpers may be introduced only where they consolidate an existing
policy that is currently embedded in a staying screen method. In particular, one
plain-value helper will prepare session run markers: it reads the unseen cache, defers
view-clear while the wake coordinator still owes delivery, clears only when this view
is displayed, and returns the marker mapping. This removes unseen/view-clear policy
from `_sync_console_native_session_tabs` without moving that DOM-rendering method.

### State ownership

The controller owns and initializes:

- `_console_fleet_survivor_timer = None`;
- `_console_fleet_unseen_cache = None`.

These are post-Wave-6-baseline private implementation details, not members of the
recorded 31-name assignable compatibility inventory. No new screen descriptor or
shadow state is added. Focused tests and production callers move to `_fleet`.
The durable mark itself remains in the existing app-level marks service; the
controller owns its cached projection and policy, not its persistence implementation.

### Staying screen responsibilities

`ChatScreen` keeps:

- `on_mount`, `on_resume`, and `on_unmount` ordering;
- `_notify_console_fleet_teardown_if_any`, because it renders app notifications;
- `_sync_console_native_session_tabs`, because it queries and paints the session UI;
- transcript-timer creation/stop policy outside the survivor-only interval;
- the screen/app logic that resolves the displayed Console composer;
- Textual worker/timer primitives exposed to the controller through wiring.

The staying methods invoke `_fleet` directly. They do not retain any of the 16 old
method names. The existing Workspace controller properties
`_console_fleet_unseen_ids` and `_console_run_marker_with_unseen` remain use-site
adapters, but wiring points them directly at `_fleet`; Workspace does not own or cache
fleet state.

## Control Flow and Ordering

### Mount and completion handoff

1. `on_mount` shows any prior teardown notice.
2. It synchronously calls `_fleet._claim_console_fleet_wake_marks()` before any timer,
   worker, activation sync, or view-clear can run.
3. Existing 0.15-second and 0.3-second mount hedges schedule
   `_fleet.consume_pending_console_fleet_completion` and
   `_fleet._maybe_start_console_fleet_survivor_tick` respectively.
4. A completion claim finds the still-open session, activates its workspace, switches
   the chat controller, then schedules the existing exclusive console-sync worker.
   Missing sessions are acknowledged and dropped; exceptions release the claim for
   retry; successful paths acknowledge exactly once.

The identical 0.15-second completion retry used by resume/activation paths is rewired
to `_fleet`, so the first available signal still performs the handoff.

### Wake coordination

The chat controller's existing view-hook slots are bound directly to `_fleet` for
user priority, conversation visibility, and delivery-start repaint. Composer changes
and workspace session-open paths poke `_fleet` directly.

The displayed-composer resolver remains late-bound and preserves the hidden/resident
screen rule: a different displayed Console contributes its composer; otherwise the
current screen's composer is used. Any non-empty draft wins ties. A delivery is in
view only when this screen is displayed and its target session is active. Delivery
start still hops through the Textual message pump before arming the transcript timer.

### Durable unseen markers

The controller caches IDs against `FLEET_UNSEEN_REVISION_ATTR`. Session-tab sync passes
plain sessions, active-session identity, and the current chat controller into the
controller's marker-preparation operation. The controller:

1. reads the cached unseen IDs;
2. identifies the active conversation;
3. checks whether `fleet_wake.has_pending` still owes it;
4. leaves the mark intact when delivery is owed or the screen is hidden;
5. otherwise clears through the existing marks service and refreshes the cache;
6. derives each run marker, with live/terminal run state outranking `SUBAGENT_UNSEEN`.

The screen only supplies the returned marker mapping to the DOM surface. Workspace
browser rows call the same `_fleet` cache and marker methods, so tab and browser
surfaces share one policy and cache.

### Survivor tick

The transcript poll's current self-stop edge calls
`_fleet._maybe_start_console_fleet_survivor_tick()`. The controller creates at most one
one-second interval through the screen-supplied timer factory and records timer
creation/stopping through the existing bookkeeping callbacks.

A beat is skipped while the faster transcript timer is active. With no controller or
no unsettled child, it stops itself first; the settled-child edge then performs exactly
one final native-console repaint. An idle Console never receives this interval.

### Teardown

`on_unmount` retains its subsystem order: video drain, transcript timer stop, fleet
timer stop, cost timer stop, then the remaining subsystem cleanup. When a chat
controller exists it awaits `_fleet._record_console_fleet_teardown(controller)`.
That method snapshots killed/surviving counts before `leave_console_runtime`, stages
notices only when the visit actually ended, and leaves overlapping successor-screen
visits silent. Notification copy remains in the next screen's presentation method.

## Error Handling

- Mount wake claiming remains fail-contained and emits only exception type metadata.
- Completion handoff errors release the exact claim and return `False`.
- Missing/partial controller doubles continue to yield safe no-ops at wake seams.
- Composer and screen-resolution failures preserve the current conservative fallback.
- Survivor liveness probe failures log metadata and return `False`, preventing a
  runaway timer.
- Teardown staging happens only after a successful, truthy runtime leave.

Moving diagnostics between owner files must not change message text, metadata fields,
exception-capture disposition, or sink topology. The governed inventory is regenerated
only after proving the delta is a content-identical owner transfer plus any independently
reconciled latest-dev changes.

## Testing and Verification

Implementation follows focused TDD; no local full-suite run is authorized.

1. Add no-mount controller tests for defaults, completion claim outcomes, mount wake
   claim, user-priority/display semantics, retry/delivery hooks, unseen cache/marker
   precedence, teardown gating, and survivor timer lifecycle.
2. Extend the Wave 6 architecture test to require all 16 methods solely on
   `ConsoleFleetLifecycleController`, zero DOM calls across every controller method,
   no sibling-controller/screen reach-through, and exact named keyword-only wiring.
3. Add mutation-sensitive checks for claim release/acknowledge, durable-mark deferral,
   late-bound composer/controller access, teardown leave gating, and final settle paint.
4. Update only focused callers/fixtures that still invoke the moved screen methods.
5. Run the directly affected fleet/wake/teardown/hidden-screen/UI-freshness tests,
   targeted Ruff lint/format, changed-module compile, `git diff --check`, and the
   persistent-diagnostic inventory gates.

The implementation is complete only when the screen contains none of the 16 moved
definitions, no production caller targets those names on `ChatScreen`, Workspace is
wired directly to `_fleet`, and the focused behavior remains green.

## Scope Exclusions

- No change to fleet coordinator, wake coordinator, run-state, or marks schemas.
- No change to notification copy, timer cadence, DOM IDs, CSS, commands, or settings.
- No compatibility method shims, generic event bus, controller registry, or base class.
- No refactor of transcript timer ownership beyond its existing handoff to the survivor
  interval.
- No cleanup of unrelated fleet tests or diagnostics.

## ADR Decision

**ADR required:** no

**ADR path:** N/A

**Reason:** This task directly implements the fleet/wake ownership boundary already
approved by the Wave 6 design and `DESIGN.md` section 7. It changes neither durable
storage nor the runtime/service contract and introduces no new architectural choice.
