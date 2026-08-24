---
id: TASK-21563
title: Retire or repair the tests that cannot fail
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-08-24 17:18'
labels:
  - testing
  - test-integrity
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A small number of tests in this suite pass unconditionally. They are not weak tests that could
miss a subtle regression; they are tests whose assertions were never written, so no behaviour
of the code under test can make them red.

They fall into three shapes. Some record their verdict by logging a success or failure line
and then returning a value, which pytest ignores — so the run is green whether the line said
"✅" or "❌". Some call the code under test inside a bare `except` that swallows everything and
assert nothing afterwards. Some are scripts shaped like test files, kept where the collector
never looks.

The cost is not the wasted seconds. It is that each one occupies the place where a real check
would go, and reports that the behaviour is covered. One of them is a path-traversal test for
browser cookie extraction.

Where the intent behind a test is recoverable, it should be recovered rather than deleted:
several of these describe real, checkable properties in their own log strings and simply never
asserted them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No test in the affected set can still pass against code that violates what it claims to check, and this is demonstrated rather than asserted
- [x] #2 Where a test's intent is recoverable it is recovered, and where it is not the removal is justified by showing the behaviour is covered elsewhere or was never covered at all
- [x] #3 Any check that a rewrite turns into a real assertion and that then fails is reported, with its cause identified as product behaviour, stale harness, or a wrong expectation
- [x] #4 Nothing is left silently red or silently skipped: anything not fixed here carries a stated reason and a tracked follow-up
- [x] #5 A test whose verification lives inside a double also asserts the double was reached
- [x] #6 The size of the wider problem is measured accurately rather than estimated from a naive scan
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm each candidate still exists and still cannot fail.
2. For each, decide recover-or-remove on evidence: does the intent describe a real property,
   and is that property covered anywhere else?
3. Rewrite the recoverable ones so every claim their log strings make becomes an assertion.
4. Prove the rewrites bite by breaking the behaviour they now check.
5. Triage whatever the new assertions expose.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Six files. Two rewritten, two deleted, one test removed, one renamed and given a body.

**The security test (AC#1).** `test_cookie_db_path_validation` patched `os.path.expanduser` to
return `/etc/passwd`, called `get_chrome_cookies` inside `try: ... except Exception: pass`, and
asserted nothing. Its premise was also not a threat model — an attacker who can patch
`expanduser` does not need a cookie bug — and the module has no path validation to pin.

The real, checkable property is narrower: both paths are derived from platform literals, and
the caller's `domain_name` reaches only a parameterized `LIKE` query, so **no caller input
should appear in any path opened**. Now observed at the Local State `open`, which every
platform branch reaches unconditionally, patched in the module's namespace (not `builtins`) so
unrelated opens are not captured, and made to raise so no real Chrome profile is read.

**Demonstrated rather than asserted.** Against a mutated product that concatenates
`domain_name` into the path: the **original test passes**; the replacement **fails on all four**
parametrizations. That is the whole difference between the two.

**The RAG UI integration module (AC#2/#3/#5).** Five tests recorded their verdicts as
`logger.success("✅ ...")` / `logger.error("❌ ...")` and ended in `return True` — which pytest
ignores — so they passed on the very paths their own `❌` strings describe as failures. They
were **running**, not skipped (5 passed here). Converted to assertions; the claims were already
written, just never enforced.

Three things the conversion exposed:
1. `test_search_failure...` looked for the notification "RAG search error". The product has
   never emitted that; it says "RAG search failed". The mismatch was invisible because the
   check was an `if` around a log line. **Fixed.**
2. An empty real search returns the sentence `"No relevant context found."`, not `None` and not
   empty framing. **Pinned**, since it is what a user sees.
3. Two tests now fail for a real reason: the hand-built app double predates the Console Library
   policy work, so `_authorize_local_results_for_prompt` discards the candidate with
   `reason=not_currently_authorized`. Marked `xfail(strict=False)` against **TASK-21564** rather
   than deleted, so the coverage stays on the record (AC#4).

**AC#5.** `test_ui_settings_reach_the_search_unchanged` previously kept its only assertions
*inside* the search double — so a double that was never called checked nothing and the test
still passed. It now asserts the invocation itself.

**Deletions (AC#2).**
- `Tests/Chat/test__zz_probe.py` — a 30-iteration timing probe ending in `assert True`,
  committed under a `_zz_` name to sort last. No recoverable intent.
- `Tests/UI/verify_command_palette.py` — 268 lines of `test_*` functions that `print()` and
  return booleans. Verified **0 nodes collected** when the directory is collected (the filename
  does not match `python_files`; naming the file directly is what made it look collected). The
  palette has **76 real collected tests** across four sibling files, so nothing is lost.
- `test_performance_summary` — printed "All performance tests completed successfully" and
  asserted nothing; a test whose behaviour was reporting success.

**Renamed with a body.** `test_core_imports_without_optional_deps` had had every import deleted
down to bare comments and ended in `assert True`. It cannot make the without-optional-deps
claim in an environment where every optional group is installed, so it is now
`test_core_modules_import` and actually imports the four. Mutation-proven: pointing one entry
at a non-existent module reddens it.

**AC#6.** A naive AST scan counted **4,413** value-returning tests. That was wrong — it
descended into helper functions defined inside tests. Counting only each test's own scope gives
**28**, and most of those are fixtures misnamed `test_*` (`test_db`, `test_app`, `test_config`),
which pytest collects as tests *and* uses as fixtures. That is a separate defect and is not
addressed here; six of the 28 were in the deleted palette file.

**Evidence.** `Tests/Web_Scraping` + `Tests/RAG` + `Tests/RAG_Search` + `Tests/unit`:
**1,438 passed, 1 failed, 22 skipped, 2 xfailed**. The single failure
(`test_fts5_match_forms_shared.py::test_plain_search_and_rag_answer_answer_the_same_inflection_miss`)
fails identically on a clean `dev` checkout — it is TASK-19642.19.1's known FTS5 parity gap.

Deleted: `Tests/Chat/test__zz_probe.py`, `Tests/UI/verify_command_palette.py`.
Modified: `Tests/Web_Scraping/test_security.py`, `Tests/RAG/test_rag_ui_integration.py`,
`Tests/RAG_Search/test_embeddings_performance.py`, `Tests/unit/test_core_imports_unit.py`.
<!-- SECTION:NOTES:END -->
