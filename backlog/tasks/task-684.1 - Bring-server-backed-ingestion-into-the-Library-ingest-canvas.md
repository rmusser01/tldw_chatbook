---
id: TASK-684.1
title: Bring server-backed ingestion into the Library ingest canvas
status: Done
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 15:35'
labels:
  - ingest
  - consolidation
dependencies: []
parent_task_id: TASK-684
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Server Sources is the only way to start a server-backed ingest, and it lives in a window the Library canvas is meant to replace. The canvas currently tells the user ingest runs on Local, with no way to target a server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A server-backed ingest can be started from the Library ingest canvas
- [x] #2 The canvas shows which backend an ingest will run on and lets the user choose when both are available
- [x] #3 The Local-only quiet line no longer appears when a server backend is configured
- [x] #4 Starting a server ingest with no server configured explains what to configure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. DONE -- pure mapping layer (Library/server_ingest_request.py, 21 tests): source + option snapshot -> the kwargs ServerMediaReadingService.submit_ingest_jobs already accepts.
2. BLOCKED ON 684.2 -- route submit through the chosen backend. A server submission has nowhere to be recorded until the registry can hold a remote job: the ingest_jobs table has no origin/remote_job_id/batch_id, its state CHECK excludes the server's cancelled state, and mark_done requires a local media_id a server submission never produces. Doing the routing first would mean writing it twice.
3. Render the backend choice in the canvas, replacing the cosmetic 'ingest runs on Local' line (build_library_ingest_state's runtime_source affects nothing else today). The selector already exists as app.media_runtime_state.runtime_backend, alongside the local/server MediaReadingScopeService pattern.
4. Gate honestly: with no server configured, explain what to configure rather than failing at submit.
5. Live pass against a configured server -- not verifiable in this environment.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Routes a Library ingest to the server when the user opts in, reusing the whole existing canvas (source picking, per-type options, preflight, queue) rather than a second UI.

Shape: a pure mapping layer (Library/server_ingest_request.py) turns a source + option snapshot into the kwargs ServerMediaReadingService.submit_ingest_jobs already takes, so the request shape is testable with no UI, no I/O and no server. app.py routes on the resolved backend; the canvas renders which backend will run and offers the switch only when the server one is genuinely usable.

Three things only a live server could have told us:
- The jobs API accepts exactly video|audio|document|pdf|ebook. Its OpenAPI types media_type as a bare string, so the real set is enforced by a runtime validator. An earlier mapping sent 'plaintext' (inferred from the retired window's form dispatch) and every plain-text server ingest would have failed validation.
- cancel_media_ingest_jobs_batch is keyword-only. It was being called positionally, and the fake took a positional parameter because it had been written to match that wrong call -- so the test validated the bug.
- Runtime policy declares media.ingestion_jobs.launch.server as required_source='server', so the service refuses the launch in local mode. The feature was unusable as first built: the switch was offered, the user could pick it, and every submission failed with 'requires server mode'. The opt-in is now necessary but not sufficient -- both the resolver and the canvas apply the gate, and an unmet precondition falls back to a local ingest (the file stays where it is) while the canvas explains why.

That last point settles a design tension. Ingest is deliberately NOT driven by the Library's browse scope: switching scope to *look* at server media must never upload a file. But policy is the authority on server actions needing server mode, so both layers now apply the same gate and the canvas cannot claim one thing while submit does another. A test asserting the opposite invariant (that a server target stays named even when unusable) was rewritten, since the mismatch it guarded against can no longer occur.

Also fixed from looking at the actual screen: the switch rendered above the status line, reading as a contradiction top-to-bottom ('Import on the server' directly above 'Imports run on this machine'). State now precedes action, pinned by a test.

Files: Library/server_ingest_request.py (new), app.py (_resolve_ingest_backend/_submit_server_ingest_job/_send_server_ingest_job), Library/library_ingest_state.py (backend + gating copy), UI/Screens/library_screen.py (backend switch).
<!-- SECTION:NOTES:END -->
