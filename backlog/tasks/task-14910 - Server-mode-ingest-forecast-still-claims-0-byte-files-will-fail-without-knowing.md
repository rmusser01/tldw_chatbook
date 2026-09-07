---
id: TASK-14910
title: >-
  Server-mode ingest forecast still claims 0-byte files will fail without
  knowing
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 01:59'
updated_date: '2026-08-11 03:36'
labels:
  - library
  - ingest
  - server
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while closing task-14827 (the server-mode forecast/receipt divergence for refused files), and deliberately left out of that task's scope.

build_ingest_forecast counts every 0-byte staged file as a certain failure on BOTH backends. On the local path that is verified: run_parse_job refuses an empty source before any write, and the local governance test asserts it. On the server path nothing verifies it -- _submit_server_ingest_job builds kwargs for the empty file and sends it, so the outcome belongs to the server, which this process cannot inspect (the same reason the forecast refuses to claim anything about server tooling).

So the server forecast makes exactly one claim it has not earned. Either the client should refuse to send a 0-byte file and fail it locally with the reason it already knows -- making the claim true by construction on both backends -- or the server forecast should stop counting empty files. task-14827's server governance test deliberately holds no empty file for this reason, and says so, which is why the gap is written down here rather than papered over.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A 0-byte file staged for a SERVER import has one outcome the forecast and the receipt agree on
- [x] #2 The server governance test in Tests/integration/test_library_ingest_flow.py covers a 0-byte file without any stubbed server behaviour deciding its fate
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Decide between the two honest options on the evidence and record the reasoning in the notes.
2. Implement the decision in tldw_chatbook/Library/server_ingest_request.py: one helper both the forecast-side predicate (server_ingest_refusal) and the submit-side builder (build_server_ingest_kwargs) consult, so the claim and the enforcement cannot drift.
3. RED tests first: the refusal predicate, the submit seam raising, the app seam recording the permanent failure with that reason, and the server governance fixture grown to hold a 0-byte file.
4. Extend Tests/integration/test_library_ingest_flow.py's server governance test with the 0-byte file its docstring deliberately omitted, and rewrite that paragraph to say why it is now knowable.
5. Update Docs/User_Guide/library/import-and-export.md if the user-visible copy changed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The decision: refuse locally, keep the count.** Of the two honest options, the client now refuses to send a 0-byte file, which makes the forecast's "will fail" a statement about code this process runs rather than a guess about someone else's machine. Four pieces of evidence, not a preference:

1. *The seam already exists.* `_submit_server_ingest_job` catches `ServerIngestUnsupported` and records an immediately-FAILED permanent row carrying the reason -- that is exactly the receipt the forecast predicts. Adding emptiness to that predicate is one more entry in an existing table, not new machinery. The alternative (stop counting) would have needed a new server-only branch in `build_ingest_forecast` plus new hedge copy, to say less.
2. *The user is better off.* A 0-byte file is almost certainly a mistake; the refusal is instant and names the file, where a round trip costs an upload and returns whatever a server makes of an empty document (quite possibly a "successful" empty row -- the exact silent-empty-import outcome task-14821 removed locally).
3. *Backend-independence.* The app now gives one answer to "what happens to a 0-byte file?" whichever target is selected. The local path already refuses one (`EmptySourceIngestError`, category `empty_source`); switching the target no longer switches the semantics.
4. *It closes the gap the task names, rather than moving it.* Option B would leave the file sent with nothing said about it, which is a smaller lie but still an unstated outcome.

**What changed** -- one new function, `empty_source_refusal` in `Library/server_ingest_request.py`, consulted by BOTH the forecast-side predicate (`server_ingest_refusal`) and the submit-side builder (`build_server_ingest_kwargs`, which raises it as `ServerIngestUnsupported`). A single helper on purpose: a promise and its enforcement stated separately is how this arc's defects have arrived. Emptiness outranks the type mapping in `server_ingest_refusal`, because the pre-flight lifts 0-byte files out of `type_groups` before classifying them -- so a 0-byte .png is counted in `will_fail_empty`, not `will_fail_refused`, and its row must give the empty reason to match. A URL is never called empty (no local size to measure), neither is an unstattable path (mirroring the pre-flight's `_statted_size`, which returns `None` rather than 0 on `OSError`), and neither is a directory -- one can stat at 0 bytes on some filesystems, and "this folder is empty" is a different diagnosis with a different recovery that the Start gate already owns.

**AC#2: the fixture grew.** `test_forecast_counts_equal_the_real_receipt_for_a_server_submission` now stages `empty.txt` -- the file its docstring deliberately omitted -- and asserts it lands FAILED, permanent, with the empty reason, and is absent from `transport.submitted`. That is the honest way to cover it: the stub decides nothing about this file, because the refusal happens before the transport is reached. The docstring paragraph explaining the omission was rewritten to say why it is now knowable.

**RED evidence.** Before the fix the extended governance test failed with `assert (3, 0, 3) == (4, 0, 2)` and the receipt `[..., ('empty.txt', 'done'), ...]` -- the app sent the empty file and the stub reported it done, contradicting the forecast's "1 empty" exactly as described. Mutation-checked both halves: neutering the builder's refusal returns that same failure, neutering the predicate reddens the three unit tests that read it.

Modified: `Library/server_ingest_request.py`, `Library/library_ingest_state.py` (docstrings only -- `will_fail_empty` now records why it is true on both backends; the `staged_total` note's claim that the buckets overlap was stale, since `analyze_path` excludes 0-byte files from `type_groups`), `Tests/Library/test_server_ingest_request.py`, `Tests/App/test_submit_library_ingest_job.py`, `Tests/integration/test_library_ingest_flow.py`, `Docs/User_Guide/library/import-and-export.md`, `backlog/docs/lessons-testing-evidence.md` (the task-14827 entry ended by naming this very omission -- it now records the follow-through: a case a governance test cannot assert is usually a claim the product should not be making).
<!-- SECTION:NOTES:END -->
