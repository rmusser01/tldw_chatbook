---
id: TASK-14827
title: Server-mode forecast still diverges from the receipt for unsupported files
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 22:30'
updated_date: '2026-08-11 02:04'
labels:
  - library
  - ingest
  - server
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing the xhigh review of the forecast arc (tasks 14820-14826), flagged by the implementing agent as out of scope for that round.

In server mode an unsupported file is forecast as `will skip`, but `build_server_ingest_kwargs` raises `ServerIngestUnsupported` and the job actually **fails**. So the forecast and the receipt disagree on the server path — the same class of defect task-14820 existed to eliminate on the local path.

This matters because it is the second server-path divergence found in one review round: the first (local tooling gaps subtracted from a server-bound forecast, making every server import read as a certain failure) was a regression the arc itself introduced and is now fixed. Both hid in the same blind spot — the governance test `test_forecast_counts_equal_the_real_receipt_for_a_mixed_folder` drives the LOCAL submit path only, so nothing asserts forecast==receipt for server mode at all.

Related, same surface: the canvas still renders local tooling warnings during a server run. Post-fix the folded summary reads "no staged file needs them", which is at least true, but the warning wall is still describing a machine that isn't doing the work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In server mode, a file the server path will refuse is forecast as a failure, not a skip
- [x] #2 A governance test asserts forecast counts equal the real receipt for a SERVER submission, mirroring the local one (the absence of this test is why two server-path divergences shipped)
- [x] #3 Local tooling warnings are not presented as blocking facts during a server-targeted import
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map what the SERVER path actually refuses vs the LOCAL path (build_server_ingest_kwargs / SERVER_MEDIA_TYPE_BY_LOCAL_TYPE / _ELSEWHERE / is_web_clip_source routing in submit_library_ingest_job) vs get_type_group's UNSUPPORTED_GROUP.
2. RED-1: unit test -- a server-mode forecast over a folder holding an image file and an unrecognised extension must count both as failures, not skips.
3. Add a refusal predicate to server_ingest_request.py that asks the same functions the submit path asks (clipper route for pages, server_media_type_for otherwise); build_ingest_forecast consults it per staged file under targets_server, into a new will_fail_refused bucket; will_skip becomes 0 on the server path.
4. Word the copy: the failure segment names 'unsupported by the server' (NOT 'nothing can read this'), and the named unsupported-files line stops promising 'will be skipped' when the server will fail them.
5. RED-2 (AC#2): governance test in Tests/integration mirroring the LOCAL one -- real analyze_path, real forecast, real submit_library_ingest_job, real ServerMediaReadingService + real request schemas, real registry/reconciler; ONLY the HTTP client is stubbed, and every stub call is validated against inspect.signature of the real TLDWAPIClient method.
6. AC#3: in server mode local tooling warnings leave warning_lines/warning_commands (so the wall and its install command stop rendering) and become one quiet advisory line saying the gap affects Local imports only.
7. Mutation-check AC#1 (revert the classification, watch the governance test go red), keep the named green suites green, update Docs/User_Guide/library/import-and-export.md + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The forecast now asks the backend it is actually aimed at, and a second governance test pins that answer against a real server submission.

**What the two backends actually refuse.** LOCAL unsupported-ness is `get_type_group(...) == UNSUPPORTED_GROUP` -- a path whose extension `detect_file_type` does not recognise -- and those files are SKIPPED. The SERVER refuses a strictly different set: everything local refuses, PLUS every type it has no media type for (raster images, deliberately left server-unmapped by task-3307), and it does NOT refuse a web page, because `submit_library_ingest_job` routes pages to the clipper before `build_server_ingest_kwargs` is ever consulted. Every one of those refusals lands as a permanent FAILED row, never a skip. So a predicate borrowed from either backend is wrong in both directions.

**AC#1.** New `server_ingest_refusal(source)` in `server_ingest_request.py` asks the same functions the submit path asks, in the same order (clipper route first, then `server_media_type_for`), and returns the exact reason string the failed row will carry. `build_ingest_forecast` consults it per FILE under `targets_server` (the refusal is a property of the source, not of its group) into a new `will_fail_refused` bucket; `will_skip` is 0 on that path because the server skips nothing. Copy: `3 will be sent to the server - 2 will fail (unsupported by the server)`. 'Unsupported by the server' rather than the local vocabulary on purpose -- half of these files (images) import fine on this machine, so 'unsupported'/'will skip' would tell a user their file is unreadable when what is true is that this destination will not take it. The named-files line follows the same rule ('1 unsupported file will fail: weird.xyz.' in server mode, 'will be skipped' locally).

**AC#2 -- the real deliverable.** `test_forecast_counts_equal_the_real_receipt_for_a_server_submission` mirrors the local governance test: real `analyze_path`, real forecast, real `_resolve_ingest_backend` routing, real `build_server_ingest_kwargs`, real `ServerMediaReadingService`, real request/response schemas, real registry and real reconciler. **The only stub is the HTTP transport** (`TLDWAPIClient`), because the network cannot run in a test. Every call to it is bound against `inspect.signature` of the REAL client method so a drifting call site fails there instead of being absorbed by `**kwargs`, and a `media_type` outside `SERVER_ACCEPTED_MEDIA_TYPES` is rejected the way the live server's validator does. It therefore proves what the app SENDS and what it refuses to send -- not what a real server does with a file it received. The fixture deliberately holds no 0-byte file: the app sends one and only the server decides, which this process cannot know and the test must not invent (raised as its own task).

**AC#3.** In server mode the tooling warnings leave `warning_lines`/`warning_commands` -- which removes the whole block the canvas renders behind `if tooling_lines` (the warning summary, the 'Copy install command' button, the 'What's missing' fold) -- and become ONE advisory line: '1 local component isn't installed - that affects imports on this machine only; this one runs on the server.' Advisory lines render as quiet notes with no warning glyph, and the glyph is what carries severity here. Verified in the real widget with a headless render probe: LOCAL mounts the summary + copy button + fold, SERVER mounts only the note.

**Evidence.** RED first on all three ACs (read and reported). Mutation check: reverting AC#1's classification made the new governance test fail on the governance claim itself -- forecast (4 sent, 1 skip, 0 fail) vs receipt (3 done, 0 skipped, 2 failed) -- restored by Edit. Green: 592 passed across the named suites (state/capabilities/server-request/preflight/App submit/integration flow/option wiring) and 197 across the adjacent ingest suites incl. `Tests/UI/test_library_ingest_canvas.py`, which is another agent's file this round and was run read-only.

**Follow-ups raised, not silently absorbed:** task-14910 (the forecast still claims a 0-byte file will fail on the server without being able to know) and task-14911 (the Start gate still uses LOCAL supported-ness, so a server-mode selection the server refuses entirely stays enabled).

**Files:** `tldw_chatbook/Library/server_ingest_request.py`, `tldw_chatbook/Library/library_ingest_state.py`, `Tests/Library/test_server_ingest_request.py`, `Tests/Library/test_library_ingest_state.py`, `Tests/integration/test_library_ingest_flow.py`, `Docs/User_Guide/library/import-and-export.md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
