---
id: TASK-684.1
title: Bring server-backed ingestion into the Library ingest canvas
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 13:07'
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
- [ ] #1 A server-backed ingest can be started from the Library ingest canvas
- [ ] #2 The canvas shows which backend an ingest will run on and lets the user choose when both are available
- [ ] #3 The Local-only quiet line no longer appears when a server backend is configured
- [ ] #4 Starting a server ingest with no server configured explains what to configure
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
Slices 1 (mapping) and 2 (routing) landed, and the mapping is now LIVE-VERIFIED against a real tldw server on :8000. The UI slice (backend selector in the canvas) is the remainder.

THREE DEFECTS FOUND ONLY BY LIVE CONTACT -- all invisible to 930 passing tests:

1. media_type mapping was wrong. The server accepts ONLY video/audio/document/pdf/ebook. It rejects 'plaintext', which this mapping sent for .txt/.md -- the commonest case -- so every plain-text server ingest would have failed validation. The value came from the legacy ingest window's own form dispatch in tldw_api_events.py, which describes that window's form, not this endpoint's contract. Crucially the accepted set is NOT in the server's OpenAPI spec: media_type is typed as a bare string and the set is enforced by a runtime validator, so submitting was the only way to learn it. Now a named constant with a guard test.

2. MediaIngestJobListResponse silently discarded pagination. It declared only batch_id/jobs with extra='ignore', so the has_more/next_offset the server really sends were dropped -- meaning the poller's pagination read a field that could never exist on the typed model. The unit test passed only because it used a dict. Fields now declared; confirmed arriving live (has_more=False, limit=10, offset=0).

3. cancel_media_ingest_jobs_batch is KEYWORD-ONLY and was being called positionally, which raises TypeError. The unit test missed it because the fake service had a positional parameter -- the fake was written to match the wrong call, so it validated the bug. Both fixed; the corrected fake now fails if the call regresses. Lesson: a fake shaped by your own assumption tests nothing.

VERIFIED END-TO-END against the live server: submit (document accepted, batch + job id issued) -> local row with origin=server and ids attached -> pending_remote_batches returns the batch -> cancel accepted (success=True cancelled=1) -> server reports 'cancelled' -> reconciler maps it to local cancelled with the reason carried through -> stop condition clears.

STILL INFERRED, NOT CONFIRMED: the 'completed', 'running' and 'failed' status spellings. The server had not picked the job up before it was cancelled, so only 'queued' and 'cancelled' were observed live. Both mapped correctly.
<!-- SECTION:NOTES:END -->
