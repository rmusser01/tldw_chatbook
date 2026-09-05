# llama.cpp manual prompt-cache snapshots

Date: 2026-09-04

Status: Written design for review; implementation not started.

Task: [TASK-31552](../../../backlog/tasks/task-31552%20-%20llama.cpp-manual-prompt-cache-snapshot-manager.md)

ADR required: yes

ADR path: [ADR-119](../../../backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md)

Reason: private file ownership, retention and deletion, and a new llama-server
management boundary. [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md)
and [ADR-036](../../../backlog/decisions/036-application-service-composition-lifecycle.md)
govern privacy and app service composition.

## 1. Purpose and agreed scope

Users can save and reload processed llama.cpp context from Models > llama.cpp.
The first release manages only a server started inside Chatbook. Snapshot names
are generated timestamps. The newest 10 complete snapshots are retained across
all models in the current user profile, with a configurable keep count.

The section is named **Prompt-cache snapshots**. Its explanation is:

> Save processed context to reuse later. Restoring does not change your conversations.

Restoring loads server cache. It does not recover messages, attachments, tool
state, a Chatbook conversation, or a reproducible generation checkpoint. Chatbook
continues sending normal message history. Matching requests may reuse restored
prefixes; no next-request routing or speedup is guaranteed.

Automatic conversation binding, automatic saves/restores on sends, external
server management, router mode, arbitrary file import/export, snapshot renaming,
pinning, and a separate in-memory slot Erase action are outside this release.
No chat payload or conversation database changes are required.

## 2. Upstream evidence

[PR 26640](https://github.com/ggml-org/llama.cpp/pull/26640) merged on
2026-08-12 as `5d9e5ac30e469d44c0a5a52556de0ead03aaa5b0`. It adds media state to
slot serialization through existing save/restore endpoints. New-format files,
including text-only saves, cannot be restored by older servers. The new server
can read legacy text-only files, but v1 imports no foreign files.

The reviewed [server implementation](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/server-context.cpp)
defers busy slot operations, clears a destination when restore fails, and exposes
optional slot metrics that may be absent before a slot has processed a task.
The [API contract](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/README.md)
uses `GET /slots` and `POST /slots/{id}?action=save|restore` with a basename in
`filename`. There is no saved-file catalog endpoint.

Media restore requires the matching model/projector configuration. SWA prefix
reuse requires `--swa-full`; Chatbook does not inject it automatically because
it affects memory use. The open [hybrid/recurrent reuse report](https://github.com/ggml-org/llama.cpp/issues/25913)
means an HTTP success is insufficient evidence of useful reuse. The open
[slot-path containment issue](https://github.com/ggml-org/llama.cpp/issues/26315)
requires private, verified directories and generated regular-file targets.

## 3. Product flow

### Setup

The launcher exposes Enable snapshots, **Off** initially, and the current
retention summary. Enabling persists through the canonical config owner and
applies to the next launch; changing it while running never rewrites a live
server's launch configuration. Disabling leaves existing snapshots on disk.

Persist `llamacpp_snapshots.enabled = false` and
`llamacpp_snapshots.keep_count = 10`. Accept integer keep counts from 1 through
1000; invalid input leaves the previous value intact. F9 Settings owns these
durable settings. Launcher controls edit the same values through that owner,
without introducing a second config writer or adding to legacy settings panes.
Explain that reducing the count applies after the next completed save.

Storage uses the effective profile directory returned by `config.get_user_data_dir()`
under `llamacpp_snapshots/`. The normal panel shows "Stored on this device" and
total size; the full read-only storage path is available in Details.

### Slot and snapshot views

Preserve the existing Start/Stop controls and lazy provider-pane behavior. A
compact section contains slots, retained snapshots, and contextual actions.
Each list has stable selection and a shared action row rather than buttons in
every cell. Use the incumbent Textual tokens and keyboard conventions.

Slots show ID, Idle/Busy/Unknown, and cached token count when reported. A missing
count is **Unknown**, never zero. Distinguish known empty from idle. Empty known
slots have Save disabled with an explanatory label; an unknown count does not
by itself forbid Save. No prompt previews or inferred conversation titles.

Snapshots show local display time, model label, source slot, logical tokens,
bytes, and configuration status. Filenames use UTC plus a unique suffix, for
example `slot-2-20260904T184233.123456Z-a1b2c3d4.bin`. The suffix is generated
from a random identifier; no user text is accepted as a path component.

Save requires no modal. Restore confirmation names the saved timestamp and
destination slot and states that current cached context is replaced and may be
cleared on failure. Select the sole idle destination automatically; with several,
prefer a known empty slot but let users select another idle destination. The
historical source slot need not exist after restart. Delete confirmation names
the timestamp and size and states permanent removal of the stored snapshot.

Success examples:

- "Snapshot saved. Removed 1 older snapshot."
- "Snapshot saved; 2 older snapshots could not be removed."
- "Cache loaded into slot 2. Matching requests may reuse it."

Saved rows remain browsable and deletable while stopped or while server features
are unavailable. Restore carries a visible disabled reason. At narrow terminal
sizes, lists stack and secondary metadata moves into selected-row details;
primary actions and error text remain reachable without horizontal scrolling.

Refresh on panel entry/re-entry, readiness transition, operation completion,
and explicit Refresh. Show "Updated <time>". Recheck immediately before an
operation. There is no continuous polling or claim of live monitoring.

## 4. Ownership and launch admission

Compose one `LlamaCppSnapshotService` at the app's dependency-ready boundary.
It owns a small HTTP client, snapshot store, operation state, and immutable launch
descriptor. The screen renders projections and requests actions. Navigation
does not cancel file or HTTP work; app shutdown closes owned resources through
the existing lifecycle. Do not add a generic service registry or provider framework.

Attach the descriptor to the exact `ServerLaunchClaim`. Capture profile root,
effective executable/model/projector identities, endpoint/authentication, relevant
cache configuration, and a unique launch ID. Every operation captures that claim;
revalidate before sending and before publishing. UI edits are never a source of
connection information after launch. A late response may finish bookkeeping for
its original operation, but cannot mark a replacement server ready or restored.

In v1 the managed endpoint is a directly launched, loopback HTTP server with the
normal API paths. Resolve optional bearer authentication from that launch's
explicit arguments/environment, including a validated local key file, and retain
credentials in memory only. TLS, custom API-prefix, router, or non-loopback
launches remain valid ordinary launches but disable snapshot management with
specific configuration guidance. This deliberately bounds the first client;
it does not silently strip advanced arguments.

When snapshots are enabled, conflicting `--slot-save-path`, `--slots`, or
`--no-slots` arguments cause actionable preflight failure. Resolve duplicate
host/port arguments, relevant aliases, and environment defaults before freezing
the descriptor; never assume the form won. Pass a captured child environment.
Unknown configuration that prevents identifying the managed endpoint disables
the manager; ordinary launch behavior remains available with snapshots off.

Use a private directory unique to this launch as `--slot-save-path` and enable
`--slots`. Separate process-running from API-ready. Before management calls,
require the claimed child still alive and successful health, properties, and slot
checks consistent with the descriptor. Refuse a pre-existing listener on the
selected port; a failed child must not adopt that listener as its own server.
Apply short bounded readiness probes off the UI event loop. Do not enable or
probe by performing a save or restore automatically.

## 5. Snapshot storage and retention

Use private regular files and versioned JSON metadata rather than a new database.
Reuse ADR-029 private-path utilities and the existing `portalocker` dependency.
No raw server response, full argv, credentials, prompt preview, image/audio body,
or conversation title enters sidecars or logs. The binary contains private model
context; it is not included in conversation sync or Chatbook export. Apply
owner-only POSIX creation to files and directories, including child-created
files, without changing process-global umask. Preserve the existing honest
Windows permission classification.

The root contains a committed snapshot area and launch-specific working areas.
Only the working area is exposed to the current llama-server. Save writes a new
unique working file. Restore stages a private copy of a committed snapshot there
before sending; this prevents retention from deleting the file the server is
reading and prevents an old server generation touching a new generation's files.
Copying is off-thread, bounded by disk availability, and never a hard link.
Expect temporary space for the in-flight file in addition to the retained set.

Metadata schema v1 contains an opaque snapshot ID, UTC creation time, monotonic
catalog publication sequence, source slot, logical token count, byte length,
binary digest, model/projector identities, relevant runtime settings and build
identity, and the metadata schema version. Use verified managed artifact digests
where already available; compute missing local digests off-thread and cache only
while verified file identity is unchanged. Do not hash large models on each refresh.
Treat sidecar contents as untrusted input when re-reading them.

The store holds a cross-process catalog lock for publication and deletion. It
resolves regular-file identities without following links and admits only its
own versioned records and generated basenames. A filename prefix alone is not
ownership. Directory scans and metadata parsing are bounded; list results are
paged when needed. Foreign, malformed, incomplete, or unsafe entries never count
toward retention and never become eligible for automatic deletion.

### Save transaction

1. Reserve one service operation and unique private working target; record its
   ownership before sending. Confirm the launch and selected slot are eligible.
2. Submit once. On a valid successful response, verify returned slot/filename,
   token and byte counts, regular-file identity, and on-disk bytes. Zero-token
   saves are reported as empty and do not publish or prune.
3. Flush the completed binary, compute its digest off-thread, and publish binary
   and metadata under the catalog lock. Allocate publication order under that
   lock before writing metadata. The atomic metadata publication is the commit
   marker; only records with both validated members are restorable.
4. While still holding the lock, remove oldest committed
   records beyond the keep count captured for this save. Never prune the newly
   committed record. Creation timestamps are display data; clock changes cannot
   make the new record the oldest. Remove only verified owned binary/metadata
   pairs, recording partial cleanup failure separately from save success.

A full disk, failed HTTP request, invalid response, hash/write failure, or
interrupted publication preserves earlier completed snapshots and performs no
retention. Metadata-first hiding/tombstoning during deletion prevents a removed
binary from remaining listed as restorable if the process crashes between steps.
The store reconciles its own interrupted publication/deletion records on entry;
it never promotes an unacknowledged binary merely because its size stopped changing.

Manual Delete is available independently of saving and affects one selected
committed record. Reading/copying that record for restore coordinates with the
catalog lock. A record deleted after a restore's private copy was made does not
invalidate that already-admitted operation. Report the deletion accurately.

## 6. Compatibility and restore

Display **Matching configuration**, **Different configuration**, or
**Compatibility unknown**. A matching model alias/path is insufficient. Compare
verified model/projector identity, server executable/build identity, effective
context/cache representation and positional settings, and adapters that affect
state. Sampling temperature/seed and transport address alone do not invalidate
cache files. Reject a destination too small for the saved logical context.

Restores with a known mismatch or missing required compatibility evidence are
disabled in v1; there is no force override. Unknown or dynamically changed
adapter/projector configuration must not be classified as matching. The planner
must enumerate supported state-affecting launch options from the chosen upstream
baseline and explicitly reject unrepresentable configurations for restore.
Save and inspection can still work when compatibility evidence is incomplete.

"Matching configuration" is a preflight result, not a portability or speed
guarantee. The server remains the binary loader and final validator. Do not parse
llama.cpp's packed binary format in Chatbook. Capability status distinguishes
slots available from multimodal persistence known/supported; old or unknown
builds get honest guidance, and HTTP unsupported responses are actionable.

## 7. Pending operations, errors, and restart

Only one server Save/Restore operation is admitted per launched server at a time.
Do not use worker replacement as cancellation of an earlier mutation. Before
Save/Restore, refresh the slot state and reject currently busy targets. The
server can still defer a request if a new generation starts after this check;
the operation UI must allow a pending state and describe the cache at execution
time, without promising an atomic idle-slot reservation.

On timeout/disconnect after submission, show **Outcome unknown** and do not retry,
publish an unacknowledged save, prune, or claim cancellation. GET refresh can
inform the display but token counts alone cannot prove a restore completed.
Keep Save/Restore disabled for that generation until completion is established
or its server is confirmed stopped. Catalog browsing and confirmed deletion
remain available because submitted restores use separate working copies.
Existing Stop remains an
explicit way to end the generation; it does not claim to preserve its cache.

After confirmed stop, discard only that operation's verified owned working
files. Following an app crash, old working areas whose server liveness cannot be
established remain incomplete and excluded from retention; they never block new
launches in distinct working areas. Show their disk usage separately and permit
cleanup only after writer termination is established. Never infer termination
from the age of a directory or a reused PID alone.

Differentiate disabled feature, still loading, authentication failure, missing
snapshot, incompatible configuration, insufficient disk/context space, restore
failure, unknown outcome, and saved-with-cleanup-failure. Use bounded safe copy;
never echo arbitrary provider error bodies. Always refresh after acknowledged
restore success/failure. A failure can leave the destination empty while the
source snapshot remains usable. Missing slot metrics remain unknown.

## 8. Implementation seams and verification

Expected implementation seams:

- `app.py`: compose and shut down the snapshot service.
- `Event_Handlers/LLM_Management_Events/llm_management_events.py` and
  `server_lifecycle.py`: launch descriptor, readiness, and generation ownership.
- A small `LLM_Management` snapshot service/store/client module set: filesystem
  transaction rules and typed management operations; independent of Textual.
- `UI/LLM_Management_Window.py`: delegate to a focused snapshot widget; preserve
  deferred views and current model source ownership.
- `UI/Screens/settings_screen.py` and `config.py`: canonical persistence.
- Targeted UI, lifecycle, transport, store, and opt-in live tests.

Before coding, read the task, place it In Progress, and add the implementation
plan with the ADR check. Split implementation work if the complete vertical
slice cannot fit a single reviewable PR; no speculative sub-framework is needed.

Required evidence includes:

1. Deterministic store tests for rapid filename allocation, clock rollback,
   retention across models, count changes, invalid/foreign files, symlinks,
   two-process publication/deletion, full disk, interrupted writes, and failed
   cleanup. Verify that no earlier snapshot is pruned on unsuccessful save.
2. Recording HTTP tests for optional/missing slot fields, readiness/auth, args
   precedence, stale launch completion, busy races, timeout without retry, and
   restore failure. Verify unsupported configurations have truthful controls.
3. Production-shaped Textual tests for keyboard Save/Restore/Delete, disabled
   reasons, preserved focus and selection, navigation away/back during operations,
   and visible controls at 80x24 and a normal wide terminal. Use the real CSS stack.
4. Isolated opt-in real-server tests: text and image save, process restart, restore,
   then send the matching request without forcing `id_slot` and measure cached
   prefix reuse. A different image must not reuse its mismatched media prefix.
   Include matching projector and any required SWA configuration; record exact
   build/model/settings. Add audio coverage when claiming a tested audio model.

Do not present mocked HTTP 200 responses, file existence, or elapsed-time-only
comparison as proof of reuse. Use server cache counters/timings with a cold
control. If hardware or a required model is unavailable, leave the live criterion
open and identify the missing evidence. Run targeted tests only; a full suite
requires the user's separate request. Live launches use isolated config and data
paths according to the repository's testing and live-verification lessons.

## 9. Review disposition

The design incorporates the independent review's seven corrections: cache-only
semantics, destructive restore failure, stale/busy observations, effective launch
identity, commit-before-prune retention, explicit compatibility, and private
snapshot provenance. Additional concrete choices are opt-in activation, positive
bounded retention, unknown-compatibility restore blocking, loopback HTTP management
for v1, and per-launch working files for uncertain operations.

These choices are written for review before implementation planning. No production
code, snapshot files, or model executions were created as part of this design.
