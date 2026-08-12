# TASK-3401.19: App-owned generated-video startup retention

## Status

Approved in conversation on 2026-08-09. This design implements ADR-044's
existing startup-retention boundary; it does not introduce a new storage or
retention policy.

## Problem

`ChatScreen._ensure_console_video_store()` currently constructs a `VideoStore`
and immediately calls `enforce_retention()`. Navigation always constructs a
fresh `ChatScreen`, so leaving Console for Settings and returning creates a new
store and reruns the `session` policy. That policy removes every generated
video, including videos created moments earlier in the same app run. Persisted
cards then render as expired even though the app has not restarted.

The lifetime mismatch is the defect: retention belongs to one application
startup, while the cleanup is owned by a replaceable screen instance.

## Decisions

### 1. `TldwCli` owns one store and one startup sweep

After `TldwCli.__init__` loads the effective application configuration, and
before any navigation screen is constructed or shown, it creates one
`VideoStore` and synchronously runs `enforce_retention()` exactly once. The
store is retained as an app-owned attribute for the lifetime of that
`TldwCli` instance.

“Once” means once per `TldwCli` instance. Constructing a new app instance is a
new real startup and must run retention again. No process-global singleton or
module-global one-shot flag is introduced.

The sweep is deliberately synchronous. The approved requirement is that
cleanup finishes before the first screen appears; a background worker would
allow Console generation or restoration to race the sweep. Startup timing may
be recorded through the app's existing startup-phase metrics, but no new
progress UI, coordinator, or dependency is required.

The `VideoStore` import remains local to the startup helper if necessary to
avoid widening `app.py`'s import graph through `Video_Generation.config`.

### 2. Fresh Console screens borrow; they never clean

`ChatScreen._ensure_console_video_store()` becomes a pure ownership lookup:

1. return an explicitly injected screen-local `_console_video_store` when a
   focused test supplied one;
2. otherwise return the running app's owned generated-video store;
3. never construct a store and never call `enforce_retention()`.

The explicit override preserves the existing narrow unit-test seam. Production
must not silently fall back to a fresh store, because such a fallback would
recreate the lifecycle bug under a different spelling. A missing app-owned
store is a wiring error and should fail loudly at the ownership boundary rather
than performing destructive cleanup from the screen.

All generation, card resolution, Play, Save, and Regenerate paths already pass
through `_ensure_console_video_store()` and therefore converge on the same
app-owned object without additional call-site changes.

### 3. Startup cleanup failure is attempted once, never retried by navigation

The app catches an unexpected startup retention exception, emits bounded
diagnostics using the operation and exception type rather than media paths or
file contents, and continues startup with the same store object. The failure is
not retried by `ChatScreen` construction or navigation.

This task does not redesign `VideoStore`'s existing best-effort per-file
deletion behavior. It changes ownership and call frequency only. Any file the
underlying store could not remove remains subject to the existing resolve and
tombstone behavior; a broader failure-policy change would require separate
acceptance criteria.

### 4. TTL semantics remain startup-scoped

`ttl` retention also runs once at app startup: stale files are removed and
fresh retained files remain resolvable across the new app instance. Navigating
between screens never reapplies age checks. Runtime configuration cache resets
do not replace the app-owned store or trigger retention again.

### 5. The post-save capacity defect is separate

Review found that ADR-044 and completed TASK-3401.4 require the total store size
cap to hold during a long-running session, while current production applies the
cap only when `enforce_retention()` is called. Moving retention to app startup
does not create that defect and TASK-3401.19 must not hide a second policy
change inside a lifecycle fix. TASK-3401.20 tracks post-save oldest-first cap
enforcement atomically.

## Lifecycle

```text
TldwCli construction
  -> load effective config
  -> construct app-owned VideoStore
  -> run retention once and finish
  -> compose/mount first destination

Console visit A
  -> fresh ChatScreen borrows app store
  -> generate video into app store

Settings visit
  -> ChatScreen A unmounts; app store remains

Console visit B
  -> fresh ChatScreen borrows the same app store
  -> current-run video still resolves ready

Next TldwCli construction
  -> new app-owned store
  -> session retention removes prior-run video
  -> persisted card resolves expired
```

## Verification

Focused tests must use temporary user-data roots and real `VideoStore` files;
a fake that merely counts calls cannot prove bytes survive navigation.

1. **Startup ownership:** constructing a production `TldwCli` with a temporary
   store runs retention once before the first `ChatScreen` is created.
2. **Real navigation:** under a mounted production app, generate or plant a
   current-run file, navigate Console -> Settings -> Console through the real
   `NavigateToScreen` path, and prove the new card spec is still `ready` and
   resolves the same bytes.
3. **Fresh-screen recreation:** construct multiple `ChatScreen` instances for
   one app and prove each returns the identical app-owned store without another
   retention call.
4. **Next startup:** create a second app instance against the same temporary
   session-retention root and prove the prior-run file is removed and its card
   becomes expired.
5. **TTL restart:** a fresh, within-TTL file survives the second app startup;
   stale TTL content is removed.
6. **Failure containment:** an injected startup sweep exception is observed
   exactly once, logs no private path or media identity, and creating multiple
   Console screens does not retry it.
7. **Mutation proof:** temporarily restore retention to
   `_ensure_console_video_store()`; the mounted navigation regression must fail
   because the current-run file is deleted.

Only the directly affected startup, Console video-message, and VideoStore test
files are in scope. Full repository and broad UI collections remain excluded by
the user's test-scope instruction.

## Scope

Expected production files:

- `tldw_chatbook/app.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`

Expected focused tests:

- a narrow production-app video-store lifecycle test under `Tests/ProductionApp/`
- `Tests/Chat/test_console_video_message.py`
- `Tests/Video_Generation/test_video_store.py` only if startup characterization
  requires a store-level assertion

Task/document artifacts:

- `backlog/tasks/task-3401.19 - Run-session-video-retention-cleanup-only-at-app-startup.md`
- the implementation plan written after this design is approved

## ADR check

ADR required: no

ADR path: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: ADR-044 already decides that session retention runs at app startup,
TTL files may survive restarts, video bytes remain file-backed, and missing
files render tombstones. This task moves an existing operation to its correct
lifetime owner without changing policy, storage format, schema, security
boundary, or dependency choice.
