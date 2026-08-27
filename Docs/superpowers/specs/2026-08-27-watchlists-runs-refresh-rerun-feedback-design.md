# Watchlists Runs Refresh and Re-run Feedback Design

## Context

The Watchlists Runs toolbar exposes `Refresh` and `Re-run source`, but the two
controls do not currently satisfy their labels:

- `Refresh` only calls `RunsPane._update_action_buttons()` and never reads the
  backend.
- `Re-run source` starts a real source check, but gives no immediate
  acknowledgement, does not expose an in-progress button state, and does not
  reload the Runs list when the operation finishes.

TASK-2309 already established the interaction contract for the equivalent
`Check now` action: acknowledge immediately, refuse duplicate work for the
same source, expose a durable busy state, distinguish terminal completion from
an asynchronous launch, and always clean up the busy state. TASK-2331 applies
that contract to Runs without introducing a second execution model.

The focused pre-change baseline is 53 passing tests across
`Tests/Watchlists/test_watchlists_runs_pane.py` and
`Tests/UI/test_watchlists_run_detail.py`, with two dependency warnings and
unrelated temporary-directory cleanup warnings.

## Goals

1. Make `Refresh` reload authoritative run rows and the selected run's detail.
2. Preserve the selected run by identity when it still exists, including when
   it has moved beyond the ordinary 100-row result window.
3. Make `Re-run source` acknowledge immediately, expose an accurate busy
   button, report an honest terminal or started outcome, and refresh Runs.
4. Prevent duplicate launches of the same backend-specific run target and,
   for local sources where both entry points share one source identity,
   prevent `Check now` and `Re-run source` from overlapping.
5. Preserve the mounted Runs snapshot across transient refresh failures and
   discard responses belonging to an obsolete backend.

## Non-goals

- Runs pagination or virtualization.
- Persistent run selection across app restarts.
- A generic application-wide action-state framework.
- Changes to run storage, schemas, service ownership, or backend APIs.
- Changing Check-now result semantics beyond the minimum shared identity and
  execution seam needed for duplicate protection.
- Refresh-success toasts; the refreshed rows and detail are the result.

## Chosen approach

Keep lifecycle ownership in `WatchlistsCollectionsScreen` and reuse the
existing loaders and Check-now outcome rules.

`RunsPane` remains a presentation and intent surface. It posts typed messages
for Refresh and Re-run, paints the busy state supplied by the screen, and never
calls a backend directly. The screen owns source-operation identity, backend
capture, workers, notifications, staged refresh publication, and selection
reconciliation.

This is preferred over pane-owned backend work because the pane is frequently
rebuilt as layouts and sections change. It is preferred over a generic action
framework because only two Watchlists entry points share this operation and
the existing Check-now path already supplies the necessary policy.

## Components

### RunsPane

`RunsPane` gains:

- a `RefreshRunsRequested` message;
- enough Re-run request context to identify the local source id or server job
  id, backend, and inert display name;
- a non-recomposing set of busy Re-run operation keys;
- in-place button painting.

The Re-run button is enabled only when the selected run carries the required
local source id or server job id and that target is not already running. A
Re-run launched from this pane changes the label to `Re-running...` and
disables the button until its backend call returns. If the same local source
is being checked from a different entry point, the button remains disabled
with accurate checking copy rather than pretending this pane initiated the
operation.

The table must not recompose merely because operation state changes; preserving
cursor, focus, and selected row is part of the existing Runs-pane contract.

### WatchlistsCollectionsScreen

The screen remains the sole state and backend owner. It gains narrowly scoped
helpers for:

- deriving one canonical launch-operation key from backend plus the
  backend-specific target id;
- registering and cleaning up accepted Check-now/Re-run work;
- recording which accepted checks originated from Re-run for accurate button
  copy;
- staging an explicit Runs refresh before publication;
- reconciling the selected run against fresh rows;
- scheduling run-detail loads through the existing `wc_run_detail` group.

The existing `_checks_in_flight` authority covers both Check now and Re-run.
Re-run origin is additional presentation state, not a second concurrency
authority.

### Existing backend controller and scope service

No new service method is required. Both `check_now()` and `launch_run()`
already converge on `WatchlistScopeService.launch_run()`. The screen uses the
existing controller operation and existing run-status interpretation.

The controller's existing `launch_run()` forwarding seam must accept both
`source_id` and `job_id`, matching the scope service it already delegates to.
Local launches pass `source_id`; server launches pass `job_id`. This is a
signature/forwarding correction, not a new backend API.

## Backend-specific launch identity

Normalized source rows expose a namespaced UI id such as
`local:subscription:5`, while normalized run rows retain raw `source_id=5`.
Comparing those values directly would allow a Check-now and Re-run of the same
source to bypass each other.

Local operations therefore derive one key from:

- backend `local`; and
- the raw source identifier, represented with the existing normalized
  subscription-id builder.

The key must be identical whether it is derived from a selected source row or
a selected local run row. This makes local Check now and Re-run share the same
duplicate guard. The existing normalizer ID builder is reused rather than
introducing a second string format.

The server run contract is different: `WatchlistRunResponse` guarantees
`job_id` and does not guarantee `source_id`, while
`ServerWatchlistsService.launch_run()` requires `job_id`. A server Re-run
therefore carries and launches by `job_id`, and its operation key is the
namespaced server watchlist-job identity. This prevents duplicate Re-runs of
the same server job. The current server source contract provides no reliable
job-to-source identity for cross-entry comparison, so this task does not claim
that a server Sources-tab Check now can be deduplicated against a Runs-tab
job launch without a new API lookup. It must not guess that two unrelated raw
integers are the same identity.

## Refresh flow

1. `RunsPane` posts `RefreshRunsRequested`.
2. The screen increments the Runs refresh generation for every accepted
   Refresh, then captures the current backend, selected run id, and that new
   generation token.
3. An exclusive `wc_runs` worker requests the latest 100 rows into local
   staged values. Nothing is published yet.
4. If the selected id is present, that fresh row becomes the candidate
   selection.
5. If the selected id is absent, the worker resolves it directly:
   - a valid record is appended after the authoritative newest-100 page so the
     selected row remains visible without disturbing that page's ordering;
   - an authoritative not-found result produces a `None` candidate;
   - a transient lookup failure aborts the refresh and preserves the mounted
     snapshot.
6. Before publication, the worker verifies that the backend is unchanged and
   the captured generation still equals the screen's current generation.
   Every newer accepted Refresh supersedes the older token immediately;
   obsolete results are discarded silently.
7. Rows and selection publish together.
8. The selected fresh record is loaded through an exclusive
   `wc_run_detail` worker. A cleared selection schedules the same grouped
   clear, invalidating any older detail request.

If the initial row request fails, the existing type-only error notification is
shown and rows, selection, detail, and action state remain unchanged. A detail
query failure does not roll back a successful row refresh; the existing run
detail failure note and toast explain that narrower failure.

"Authoritative not found" is intentionally narrow:

- local `get_run()` raising `KeyError`; or
- server `get_run()` raising `APIResponseError` with `status_code == 404`.

All other exceptions, including authentication, connection, timeout, policy,
validation, and non-404 server response errors, are transient/indeterminate
for this purpose and retain the complete mounted snapshot.

## Re-run flow

1. `RunsPane` posts the selected run's local `source_id` or server `job_id`,
   backend, and source/job title.
2. The screen derives the backend-specific operation key and rejects the
   request when:
   - the required target id is invalid;
   - the request backend is no longer the active backend; or
   - that operation key is already in `_checks_in_flight`.
3. An accepted request is registered in the shared in-flight set and in the
   Re-run-origin presentation set.
4. The screen immediately sends `Re-running <name>...` with `markup=False`
   and updates the mounted pane in place.
5. A named coroutine worker launches the run against the captured backend.
6. The result is interpreted using the existing Check-now rules:
   - terminal local success: `Re-run complete` with available counts;
   - queued/running server response: `Re-run started`;
   - entirely skipped result: a stated warning;
   - returned failed status: a stated error;
   - raised exception: a stated error and warning log.
7. `finally` removes both shared and Re-run-origin state and repaints the
   button, even on unexpected failure or screen/layout changes.
8. Completion dispatches an authoritative Runs refresh into `wc_runs`; it
   does not call the loader inline from the mutation worker.

Different targets may execute concurrently. Repeated local work for the same
source is refused regardless of whether it originated from Sources, Inspector,
or Runs. Repeated server Re-runs for the same job are refused. For server
responses, busy state ends when the launch request returns; the UI does not
claim to track remote execution to completion.

## Backend and stale-result safety

The backend is captured at the moment each action is accepted. No worker reads
`self.runtime_backend` later to decide where its already-accepted operation
belongs.

Refresh publication and Re-run UI cleanup are keyed by the captured backend.
Switching backend while work is in flight cannot publish local rows into a
server view, compare local and server ids, or leave the new backend's Re-run
button disabled by an unrelated operation.

Run-detail publication continues to require that the pane's current selected
run id matches the record being painted. The grouped detail request adds the
missing cancellation boundary so an older query cannot repopulate mirrors that
a later selection or clear now owns.

## Error and trust handling

- Source/job titles are user- or remote-derived and always reach notifications
  with `markup=False`.
- Unexpected exception text is not exposed where it can contain URLs or local
  paths; existing Watchlists logging and type-only notification rules apply.
- A backend switch or superseded generation is normal stale work and is
  discarded without an error toast.
- Missing backend-specific launch identity disables Re-run rather than
  launching a knowingly invalid request.
- Refresh never clears a working snapshot merely because a replacement read
  failed.

## Testing strategy

Implementation follows strict red-green-refactor cycles. Focused tests use
event-gated fakes rather than timing sleeps.

Pane coverage proves:

- Refresh posts the new message rather than repainting buttons locally;
- missing backend-specific launch identity disables Re-run;
- accepted Re-run state disables and relabels the button without recomposing
  the table;
- external Check-now activity is disabled with accurate non-origin copy.

Mounted screen coverage proves:

- Refresh reads changed backend rows;
- selection and fresh detail survive by run id;
- a selected run outside the 100-row page is fetched and pinned;
- authoritative deletion clears selection/detail;
- transient list or pin failure retains the complete mounted snapshot;
- backend-switch and superseded responses do not publish;
- detail clearing/loading uses the grouped request boundary.

Re-run coverage proves:

- immediate acknowledgement and busy state precede backend completion;
- local Check now and Re-run refuse duplicate work for the same canonical
  source, while server Re-runs use and deduplicate by the required job id;
- different sources remain independent;
- local terminal, server-started, returned-failure, skipped, and raised-failure
  outcomes are honest;
- cleanup always restores the button;
- completion dispatches an authoritative Runs refresh.

Controller/scope forwarding coverage additionally proves that local Re-run
passes `source_id` and server Re-run passes `job_id`; restoring the previous
source-only controller signature must make the server test fail.

Final verification is limited to affected Watchlists/UI tests, modified-file
Ruff, and `git diff --check`, following the user's established constraint.

## ADR check

ADR required: no

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: this task repairs two controls within the existing Watchlists screen,
pane, controller, and worker boundaries. It does not change storage, service
ownership, backend contracts, or long-lived navigation architecture.
