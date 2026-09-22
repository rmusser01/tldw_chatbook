---
id: TASK-32901
title: "Work stream: the tier-2 P2 findings"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 115 P2 findings, grouped by package family rather than by defect class so each PR stays reviewable
and one reviewer can hold the whole diff. Every member was re-validated against `origin/dev d0face3ebe`
before filing; the per-finding verdict, current file:line and the literal command that proves it are in
`qa/tier2-code-review-2026-09-21/validation/`.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every P2 is either fixed, or closed with a recorded reason
- [ ] #2 No P2 PR mixes package families
- [ ] #3 Findings the validation pass marked WRONG are closed as such, not fixed
<!-- AC:END -->

## Implementation Notes — first 44 of 115

<!-- SECTION:NOTES:BEGIN -->
Two batches, both complete, neither pushed.

**`fix/tier2-p2a`** — S01/S02 Notes, S05 Library, S12 Media (20 findings), 3 commits.
**10 fixed / 7 closed-with-reason / 3 not-real.**

**`fix/tier2-p2b`** — S06 tldw_api, S14 Web, S11 Evals+ingest (24 findings), 3 commits.
**14 fixed / 10 closed-with-reason.**

Both were briefed that the bar for *changing* code is higher for a P2 than a P0, because a speculative fix
to working code is a net loss. The resulting ratio — 24 fixed, 17 closed with a recorded reason, 3
retracted — is the intended outcome, not under-delivery.

## Three findings retracted as NOT REAL, and one of them was dangerous

- **S02.2** — the recommended correction is **actively harmful**. Adding `AND deleted = 0` to the keyword
  collision scan raises `sqlite3.IntegrityError: UNIQUE constraint failed` against two deliberately-pinned
  test cases, because `keywords.keyword` is `UNIQUE NOT NULL COLLATE NOCASE` — a soft-deleted row still owns
  its name. The difference the review called drift is **required**. Demonstrated by applying it, watching it
  fail, and reverting; a no-op guard plus an invariant comment now stops the next reader re-making it.
- **S05.4** — claims `chunk_rows` is consumed only as three aggregates. It is used at
  `local_media_chunk_tool_service.py:447` and `:480` as well. Load-bearing.
- **S05.3** — the proposed `list_review_set_summaries()` is not implementable: the consumer needs per-item
  liveness resolved against the Media DB, which `COUNT(*)/SUM(done)` cannot produce.

## Two stated mechanisms corrected

- **S02.3** — the sink at `Logging_Config.py:609` is a **function** sink; `_forward_loguru_to_standard`
  takes `record = message.record` and rebuilds a stdlib record from its fields, so `format=` never reaches
  the output. Adding `{extra}` there is a **no-op**; the one-line fix the finding implies does not exist.
  The real fix is inside the forwarder and must skip keys colliding with `LogRecord` attributes.
- **S01.4** — the stated security risk cannot occur: the lexical `_overlaps` has exactly one production
  caller, which immediately re-runs the same three comparisons with inode `samefile`. D4 hygiene, not a hole.

## Counts corrected

S06 cites 3 raw `{server_id}` interpolations; there are **57** (which is why the fix is a central endpoint
guard, not site-by-site quoting). S11's "~9,000 lines" is **7,569** — it double-counted `eval_templates.py`,
which a sibling finding says can never execute. S12.1 is **8** tables not 9, **31** call sites not 32.
`client.py` measures 16,687 not 16,661. S14's "one test hitting `searx.be`" is real but inert —
`testpaths = ["Tests"]` means none of those 21 embedded functions is ever collected.

## Notable fixes

A live XSS in `Web_Server/artifact_share_server.py:355` (`item.key` interpolated raw into an `href`; a
crafted key renders `onmouseover="alert(1)"`) — rated `inferred`, actually verified, though exploitation
still needs a tampered manifest since `new_artifact_key()` mints 128-bit opaque keys. Fixed at the render
**and** with a fail-closed `pattern` on `SharedArtifact.key`. A `DELETE … IN (…)` purge that exceeded
SQLite's variable limit, now chunked. A shadowed 1,298-line module deleted with a repo-wide guard added —
**guard verified to bite**: restoring the file produces
`these modules can never execute -- the same-named package beside them wins every import`.

## Coordination

Two known conflicts, both trivial and both flagged before merge: p2b's shared `_raise_api_error_from`
necessarily closes S06's first P1 as well (keep one copy in `_sse_request`/`_stream_request`), and p2a's
`raw_participants._file` fsync is the same root-cause fix TASK-32896 reached from the Backup_Recovery side.

**Remaining: 71 of 115 P2 findings**, in the slices not covered by these two batches.
<!-- SECTION:NOTES:END -->
