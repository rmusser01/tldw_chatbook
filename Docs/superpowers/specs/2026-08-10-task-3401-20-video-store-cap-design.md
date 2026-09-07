# TASK-3401.20: Enforce generated-video capacity after every save

## Status

Approved in conversation on 2026-08-10 after review of the ordinary-save,
oversized-result, failure, and overwrite paths. This design amends ADR-044's
capacity policy; it does not change the message schema or make generated video
durable by default.

## Problem

`VideoStore.enforce_retention()` applies the configured total-size cap only
when the application starts. `VideoStore.save()` writes a generated video but
does not enforce that cap. A long-running Console session can therefore grow
the ephemeral store without bound even though ADR-044 says the cap applies in
both retention modes.

Blindly invoking `enforce_retention()` after every save is not safe. In the
default `session` mode that method intentionally removes every pre-existing
video because it is the startup sweep. Calling it during a run would expire
every earlier card whenever a new video is generated.

A second boundary case also needs an explicit product decision: one generated
video may itself exceed the configured cap. Silently deleting other videos,
silently discarding the new result, or placing an over-cap file in the managed
store without consent would each violate a different user expectation.

## Decisions

### 1. Ordinary saves enforce capacity at the storage boundary

`VideoStore.save()` is the single boundary for a normal generated-video save.
When the new payload is no larger than the configured cap, the store:

1. takes a per-store re-entrant lock and the root's interprocess capacity
   lease;
2. writes the payload to a private sibling stage and atomically publishes the
   complete new target while all prior managed files remain intact;
3. snapshots the actual managed files and their current sizes;
4. chooses the oldest existing files required to make room, using file
   modification time with a deterministic path tie-breaker;
5. removes those victims while the generation caller still retains the source
   bytes needed to create a recoverable pending artifact on failure; and
6. verifies the resulting managed total is at or below the cap before
   reporting success.

The new file is never an eviction candidate in its own save. Because a normal
payload individually fits the cap, removing enough existing files always makes
room when the filesystem permits it. Existing cards for removed files resolve
as expired through the current tombstone behavior; the newly appended card and
unrelated survivors remain ready.

The in-process lock belongs to each `VideoStore` instance. The application
deliberately permits multiple app instances to share one profile, so every
capacity-changing operation also takes a root-scoped exclusive `portalocker`
lease on a stable sibling lock file. Acquisition uses one bounded monotonic
timeout (five seconds in production), never an unbounded wait. `portalocker` is
already a core dependency and proven repository primitive; this adds no
dependency or process-global coordinator. Startup retention, ordinary saves,
and explicit oversized adoption use the same lock order and never reacquire
the interprocess lease recursively. Two distinct `VideoStore` instances,
including instances in separate processes, therefore cannot both snapshot and
commit against the same root concurrently.

Timeout raises one typed store-busy error. At startup, the existing
`_build_generated_video_store()` containment logs the exception type and
continues boot without a sweep. During generation or managed adoption,
`run_video_generation()`/the outcome resolver retain the staged bytes and
present the recoverable `store_failure` choices instead of blocking the UI or
losing the result. Focused tests shorten the module timeout through a narrow
test seam; there is no user-facing timeout setting.

Read snapshots and resolution do not expose a partially written target. The
new managed target is published before destructive eviction so a publication
failure leaves every old managed video untouched. If a later victim deletion
fails, the store removes the just-published target and raises a typed
managed-save failure while `run_video_generation()` still owns the adapter
bytes and can stage them for the caller. Some earlier victims may already have
been removed; they were valid oldest-first victims, but the remaining managed
store is never knowingly left over cap and the generated result is still
recoverable.

### 2. Inventory and eviction never follow links out of the store

Capacity enforcement turns inventory into a destructive path, so it must not
reuse the current follow-link behavior. Inventory uses non-following directory
metadata, accepts only safe-component real message directories, and accepts
only regular files directly inside those directories. Symlinked directories,
symlinked files, Windows reparse points/junctions, nested directories, and
entries whose resolved parent is outside the resolved store root are excluded
from managed capacity victims.

Immediately before each unlink, the store repeats non-following type,
reparse-point, safe-component, and resolved-root containment checks. A failed
check aborts that save/adoption without deleting the entry. This task does not
attempt to clean suspicious entries; it fails closed and leaves them for the
user. Diagnostics contain only the operation and exception/rejection type,
never the external target or private path.

Focused tests plant a symlinked message directory and a link/reparse-style file
where the platform supports them, then prove ordinary save, startup capacity,
and evict-all never modify the external target. Platform-specific reparse tests
may be skipped only when the host cannot create that entry type; the POSIX
symlink escape remains mandatory on POSIX.

The policy remains **oldest-by-mtime**, matching the existing implementation
and task wording. This task does not add access tracking or turn the store into
a true least-recently-used cache.

### 3. Runtime capacity enforcement is separate from startup retention

The store gains one cap-only internal operation. `save()` and the capacity
portion of `enforce_retention()` may share that operation, but `save()` never
calls the full retention method and never evaluates `session` or TTL age.

`enforce_retention()` remains the app-start boundary:

- `session` removes prior-run videos;
- `ttl` removes only expired videos; and
- capacity is applied to the survivors.

This separation is load-bearing. A post-save mutation that calls the startup
sweep must fail a focused test with a current-run video that is not required as
a capacity victim.

### 4. An unstored video is an expected generation outcome

When `len(content)` exceeds the configured cap, `VideoStore.save()` writes no
managed file and returns a typed capacity outcome containing only the payload
size and configured capacity. Capacity overflow is not represented as an
unexpected adapter failure and does not append a Console message. A filesystem
failure during an ordinary managed save is still an error, but the generated
payload is preserved for user recovery instead of disappearing with the
exception.

`run_video_generation()` builds the normal `VideoGenerationMetadata`, writes
the generated bytes into an auto-deleting standard-library `TemporaryFile`,
and returns a `PendingVideoArtifact` when the result remains outside the
managed store. The artifact records whether the reason is `over_capacity` or
`store_failure` and owns:

- the metadata and preallocated stable message id;
- the payload and cap sizes needed for user copy;
- the generated slug and extension; and
- the temporary binary handle plus an idempotent `close()` operation.

The adapter currently returns bytes in memory, so staging cannot remove that
initial materialization. It does release the potentially hundreds-of-megabytes
payload as soon as the worker returns instead of retaining it throughout the
user decision. The temporary handle has no durable application path and is
closed on every terminal path, including cancellation or screen teardown.

Normal generations keep the existing `(metadata, managed_path)` return shape.
The generation result becomes the narrow union of that successful shape and
`PendingVideoArtifact`; provider adapters remain unaware of storage policy.
For a normal publication or eviction failure, `run_video_generation()` stages
the still-live adapter bytes before allowing the worker result to unwind. That
artifact uses the same external-save/discard ownership path and offers Retry
here instead of the oversized result's Evict all action.

### 5. The Console offers exactly three oversized-result choices

Initial generation and Regenerate both pass their result to one asynchronous
Console outcome resolver. An oversized artifact opens a dedicated modal that
states the generated size, configured cap, and the consequences of each
choice:

1. **Keep here — remove other videos.** Evict every currently managed video,
   then adopt the staged result as the store's sole file and append its normal
   Console video card.
2. **Save to disk.** Open `EnhancedFileSave` with `<slug>.mp4` as the default,
   copy the staged bytes to the selected destination, open the completed file
   with the operating-system player, and append no Console video card.
3. **Discard.** Close the staged artifact and append no Console video card.

Dismissing the choice modal or cancelling the file picker is equivalent to
Discard. The generated command draft is not restored after a deliberate
choice: generation succeeded and the user selected the result's disposition.

Before opening a modal, the resolver registers the artifact in a screen-owned
pending-artifact dictionary keyed by its preallocated message id. Its
`try/finally` unregisters and idempotently closes the artifact after successful
managed adoption, successful external save, explicit discard, picker
cancellation, or task cancellation. `ChatScreen.on_unmount()` atomically drains
and closes any artifacts still registered, so navigation or app exit while the
choice modal or file picker is open cannot orphan the stage. Resolver cleanup
may run afterward and remains safe because `close()` is idempotent.

A `store_failure` artifact uses the same owner and dialog surface with three
accurate actions: Retry here, Save to disk, or Discard. It never claims that
capacity was exceeded and Retry uses the normal capped save path rather than
evict-all.

### 6. Managed oversized adoption is a sole-file cap exception

“Keep here” is the user's explicit approval for a narrow exception: the new
video may exceed `max_store_mb` only while it is the sole managed file.

Under both store locks, adoption publishes a complete candidate while retaining
the external stage, then removes all prior managed videos. If publication
fails, every old video remains. If any old file cannot be removed, the store
withdraws the candidate and appends no card. Some already-selected old victims
may have been removed—deletion of those files was the action the user
approved—but the staged new result remains available and the choice modal is
offered again.

If candidate publication fails, eviction never starts and the staged artifact
remains available for retry, external save, or discard. The application never
claims that the new card is ready until publication and all required eviction
complete and the managed file is resolvable.

No persistent exception flag is required. A sole oversized managed file is
recognizable from the store contents themselves:

- a fresh sole oversized file survives `ttl` startup capacity enforcement,
  subject to the existing TTL age rule;
- `session` startup retention still removes it;
- a later ordinary save evicts it after publishing the new candidate but
  before reporting that in-cap save as successful; and
- if corruption or an external race leaves additional managed files beside an
  oversized file, ordinary oldest-first enforcement restores the cap rather
  than preserving a multi-file exception.

### 7. Deletion and publication failures fail closed

The current best-effort startup sweep may continue past an individual unlink
failure, but a save must not subtract a victim's recorded size before proving
that the file was actually removed. A normal save reports success only when
the new file exists and an actual post-operation snapshot is within capacity.

For an ordinary save, the complete new target is published while all old files
remain. Publication failure leaves them untouched. If a required victim cannot
be removed, the new target is withdrawn and the staged generated result is
returned as a `store_failure` artifact. For oversized adoption, any incomplete
evict-all attempt similarly leaves the staged result outside the managed
store. Failures are surfaced to the user and logged with bounded operation and
exception-type fields, not prompt text, media bytes, staged filenames, or
private filesystem paths.

An action failure does not immediately destroy the pending result. The
Console reports the failure and offers the three choices again while the stage
is still valid.

### 8. External save is atomic and never silently overwrites

`EnhancedFileSave` currently returns an existing destination without an
overwrite check. The oversized external-save flow therefore performs an
explicit confirmation when the selected target already exists. Declining that
confirmation returns to the file picker while retaining the staged result.

After a new target or confirmed replacement is selected, copying runs off the
UI thread into a complete temporary sibling. A destination that did not exist
at selection is committed with an atomic no-clobber operation; if another
writer creates it first, the app overwrites nothing and returns to the
confirmation flow. For a confirmed replacement, the app records the target's
non-following identity before confirmation and revalidates that identity
immediately before atomic replacement. A missing or changed identity requires
fresh confirmation. Portable filesystems do not provide a universal
path-version compare-and-swap after that final check, so confirmation grants
permission to replace that named path; the no-clobber path still guarantees
that a never-confirmed destination is not overwritten. Any temporary sibling
is removed in `finally`.

Successful external copy closes the staged artifact and leaves no managed file
or Console card. The app then launches the saved path with the platform's
existing `open`, `os.startfile`, or `xdg-open` behavior. A launch failure does
not delete the successfully saved file; the user is told that the save
succeeded but automatic opening failed.

### 9. Existing boundaries stay intact

- Video bytes remain outside the database.
- Message metadata and stable-id persistence do not change.
- Provider adapters and ComfyUI workflows do not change.
- The configured cap remains in MiB using the existing minimum and defaults.
- No new dependency, background service, global singleton, schema migration,
  access-time index, or general-purpose storage abstraction is introduced.

## User flow

```text
generation worker returns bytes
  -> VideoStore.save
       -> payload <= cap
            -> stage and publish complete new file under both locks
            -> evict required oldest files
            -> return metadata + managed path
            -> append normal Console card
            -> publication/eviction failure
                 -> withdraw new managed target
                 -> stage adapter bytes in PendingVideoArtifact(store_failure)
                 -> Retry here / Save to disk / Discard
       -> payload > cap
            -> no managed write
            -> stage in TemporaryFile
            -> return PendingVideoArtifact(over_capacity)
            -> show three-choice modal
                 -> Keep here
                      -> publish complete candidate under both locks
                      -> evict all prior managed files
                      -> commit candidate as sole exception
                      -> append normal Console card
                 -> Save to disk
                      -> picker + overwrite confirmation when needed
                      -> atomic external copy
                      -> open with OS player
                      -> no Console card
                 -> Discard/cancel/unmount
                      -> close stage
                      -> no Console card
```

## Verification

Only tests related to touched files are authorized. Focused evidence must cover
the real storage and Console seams rather than a fake that repeats the intended
call shape.

1. **Ordinary post-save cap:** real temporary files prove every successful
   normal save leaves actual bytes at or below the cap, evicts oldest-by-mtime,
   expires the old card resolution, and preserves the new file.
2. **No startup-policy reuse:** under `session` and `ttl`, an in-cap save does
   not remove a non-victim current-run file or evaluate its age.
3. **Sole oversized exception:** explicit adoption leaves exactly one managed
   file above the cap; fresh TTL startup retains it, session startup removes
   it, and the next normal save evicts it.
4. **Concurrent saves:** controlled saves through two `VideoStore` instances
   and at least two independent processes sharing one real root cannot both
   bypass capacity; the final actual total is bounded and every
   reported-success path was complete when returned.
   A held independent lease also proves acquisition times out within the
   configured bound: startup containment continues and generation returns a
   readable `store_failure` artifact rather than hanging.
5. **Failure honesty:** injected unlink and publication failures never append a
   new card, never report a cap-compliant save falsely, leave all old files
   intact on publication failure, retain the new generated bytes as a pending
   artifact, and emit sanitized diagnostics.
6. **Pending artifact ownership:** the worker returns exact metadata and bytes
   through the temporary handle for both over-capacity and managed-save
   failures, writes no falsely successful card, and closes the handle on every
   terminal UI path.
7. **Initial and Regenerate parity:** both real dispatch paths use the shared
   resolver and drive Keep, Save to disk, and Discard without bypasses.
8. **External save:** the picker defaults to `<slug>.mp4`; cancellation makes
   no card; existing targets require confirmation; decline preserves the
   stage; a concurrent creator is not overwritten; a changed confirmed target
   is reconfirmed; exact bytes reach the destination; success opens the saved
   path; and launch failure preserves it.
9. **Modal action recovery and teardown:** managed-adoption and external-copy
   failures show guidance and re-offer choices while the artifact remains
   readable. Mounted navigation/app-exit tests close registered artifacts while
   both the choice modal and file picker are open, with late resolver cleanup
   remaining harmless.
10. **Non-following destructive scan:** real external sentinel files behind a
    symlinked message directory and link/reparse-style file remain byte-exact
    after ordinary capacity eviction, startup capacity, and evict-all.
11. **Mutation proofs:** removing the save-time capacity hook, reversing victim
    order, removing either lock, following a linked directory, calling full
    retention from `save()`, bypassing no-clobber/overwrite confirmation, or
    closing the artifact before a retry must each fail a named focused test.

The focused commands may include only the changed VideoStore, Console video
generation, and new capacity-dialog test files plus the single existing
production-app startup-containment test reached by the new lease timeout. Full
repository, broad Chat/UI, RuntimePolicy, and unrelated generation suites
remain excluded by the user's test-scope instruction. No live generation
server is required because this task changes the post-adapter storage and UI
outcome boundary; provider live UAT already exists independently.

## Expected scope

Production:

- `tldw_chatbook/Video_Generation/video_store.py`
- `tldw_chatbook/Chat/console_generate_video.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- one narrow Console oversized-capacity modal under `tldw_chatbook/Widgets/`

Focused tests:

- `Tests/Video_Generation/test_video_store.py`
- `Tests/Chat/test_console_generate_video.py`
- one narrow Console oversized-capacity flow test
- the targeted generated-video startup-containment case in
  `Tests/ProductionApp/test_chat_composition_retirement.py`

Documentation:

- this design
- the TASK-3401.20 backlog record and implementation plan
- ADR-044's capacity-policy amendment

## ADR check

ADR required: yes

ADR path:
`backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: this design changes ADR-044's previously absolute managed-store cap by
adding a user-approved sole-file exception and introduces a typed
generation/storage/UI capacity outcome. Those are durable storage-policy and
cross-module contract decisions. Amending the existing ADR is clearer than
creating a second decision for the same ephemeral-video boundary.
