---
id: TASK-15782
title: Repair the generic ingest-options snapshot test asserting a stale hardcoded dict
status: Done
assignee:
  - '@claude'
created_date: '2026-08-13 12:31'
labels:
  - test-health
  - library
priority: low
---

## Description

Found and flagged as pre-existing and unrelated in task-15470's
Implementation Notes (input-latency burn-down's config-persistence task):
`test_options_persist_to_config` fails on a content mismatch — a schema
drift, not a threading bug. Task-15470's notes record that this failure was
"exposed only once the `run_worker` crash this task fixed stopped masking
it," meaning the test has likely been silently failing-or-skipped-via-crash
for a while and nobody noticed once the underlying worker exception stopped
swallowing it. The test itself asserts against a hardcoded dict of expected
generic-ingest options that has drifted out of sync with whatever the ingest
options form/save path actually persists today.

## Acceptance Criteria

- [x] `test_options_persist_to_config`'s expected dict is reconciled against
      current production behavior — diagnosed as either a genuinely stale
      test expectation (update the test) or a real regression in what gets
      persisted (fix production and keep the test's original intent)
- [x] The specific drifted key(s)/value(s) are identified and documented in
      the task notes, not just "test updated"
- [x] The test passes on dev without weakening its coverage (it should still
      catch a genuine future persistence regression, not just always pass)

## Implementation Plan

1. Re-locate `test_options_persist_to_config` at HEAD (`Tests/integration/
   test_library_ingest_flow.py`) and run it to see its current pass/fail
   state -- task-15470's notes describe the failure as it stood on an
   earlier commit, not necessarily HEAD.
2. `git log -p` the test to find whether/when the "content mismatch" was
   already patched, and what the patch actually did.
3. Trace production's `_build_ingest_options_snapshot`
   (`tldw_chatbook/UI/Screens/library_screen.py`) to find the real source
   of truth for the "generic" group's persisted defaults.
4. Diagnose: test-side drift vs. a genuine production regression (silent
   field drop, the task-3309 class).
5. Rewrite the test's expected dict(s) to derive from that source of truth
   instead of literal values, keeping explicit literals only for the
   form-submitted overrides the test means to exercise.
6. Mutation-verify in both directions: (a) temporarily make production
   drop one schema field -- confirm the test goes red; (b) temporarily add
   a brand-new schema field -- confirm the test stays green with no test
   edit (drift resistance). Restore both probes via Edit.
7. Run the full test file + sibling Library ingest test files; ruff check
   the touched file.

## Implementation Notes

**Diagnosis: test-side drift, not a production regression.** At this
worktree's HEAD (`0b937e31d`), `test_options_persist_to_config` already
PASSES -- the "content mismatch" task-15470 flagged had already been
reactively patched in commit `0acc6eeeb8` ("test(library): isolate ingest
capability fixtures", 2026-08-14), which hand-added the seven fields the
test's expected `generic` dict was missing (`overwrite_existing`,
`custom_prompt`, `system_prompt`, `generate_embeddings`,
`keep_original_file`, `chunk_overlap`, `encoding`). But that fix was
*itself* another hardcoded literal dict -- the same class of drift that
caused the original mismatch, primed to rot again the next time a
`generic` field is added or a default changes.

**Root of the original drift:** the "generic" capability schema
(`tldw_chatbook/Library/ingest_capabilities.py`, `_TYPE_GROUPS["generic"]`)
grew past what the test's literal dict tracked across two commits --
`chunk_overlap`/`encoding` existed since the original ingest feature
(`40fe66c01`, PR #717) and `overwrite_existing`/`custom_prompt`/
`system_prompt`/`generate_embeddings`/`keep_original_file` were added in
`2e2a745e4` ("feat(library): add mode-aware ingest capabilities") -- while
the test's dict, first written at the task-3300 era, only ever asserted
`analyze`/`chunk`/`chunk_size`. Task-15470's notes record this being
*discovered* rather than *introduced*: an unrelated `run_worker` crash on
the unmounted test screen was throwing first and masking the downstream
content assertion; once task-15470 fixed that crash (patching
`_save_library_ingest_options` instead of the module-level config
function), the pre-existing content mismatch became the visible failure.

**Fix:** rewrote `test_options_persist_to_config`'s expected `generic` and
`ebook` dicts to derive from the same schema production reads
(`ingest_capabilities.get_capabilities("generic"/"ebook").fields`) instead
of hand-copied literals, mirroring exactly what
`_build_ingest_options_snapshot` does
(`generic.setdefault(field.name, field.default)` for every schema field;
the `ebook.chunk_method` schema-lookup-with-"chapters"-fallback). The three
fields the test's own submitted form overrides (`analyze`, `chunk`,
`chunk_size`) are still asserted as explicit literals, since exercising a
submitted value winning over the schema default is the actual behavior
under test there. The expectation is built from the schema *dataclass*,
never by calling `_build_ingest_options_snapshot` itself, so the assertion
stays a real check rather than a tautology.

Mutation-verified both directions (temporary edits, restored via Edit
before committing):
- Made `_build_ingest_options_snapshot` skip `generate_embeddings` (task-
  3309-class silent drop) -- the test correctly went red.
- Added a brand-new `OptionField` to the "generic" schema -- the test
  stayed green with zero test edits, confirming the drift-resistance goal.

**Tests:** `Tests/integration/test_library_ingest_flow.py` (13 passed) and
the sibling Library ingest surfaces `Tests/UI/test_library_ingest_canvas.py`
+ `Tests/UI/test_library_ingest_retry_last.py` (161 passed combined, no
regressions). `ruff check` clean on the touched file; `ruff format --check`
flags pre-existing, unrelated formatting drift elsewhere in the same file
(confirmed present before this change via a stash/pop A-B) -- left
untouched since touching it would inflate this diff with unrelated churn.

**Modified files:** `Tests/integration/test_library_ingest_flow.py` only
-- no production change; this was a test-side fix.
