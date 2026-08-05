---
id: TASK-1261
title: Make the NLTK unusable-tokenizer test independent of the installed extras
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 18:51'
labels:
  - bug
  - testing
  - chunking
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Utils/test_startup_polish_regressions.py::test_nltk_download_false_is_not_logged_as_success` fails on any machine where the optional `nltk` package is not installed, and the way it fails points at the wrong thing.

The test simulates "nltk is installed but its tokenizer data is unusable" by setting `Chunk_Lib.NLTK_AVAILABLE = True` and stubbing the probe and download helpers. But `_ensure_nltk()` still performs a real `import nltk` (Chunk_Lib.py:139). Without the package that import raises, and the function returns at line 142 — before reaching the warning at lines 153-158. The test then reports "no WARNING/ERROR mentioning punkt was logged", which reads exactly like the production warning has been lost.

It has not been lost: with `nltk` installed the test passes and the warning fires correctly. The defect is that the test silently depends on an optional extra (`chunker`/`websearch`) while claiming to exercise logic independent of it, and emits a misleading failure signal when that extra is absent.

Note: this task was originally filed on the opposite, incorrect diagnosis — "the warning was lost in a refactor". That conclusion came from probing in the same nltk-less environment, which hits the same early return. Corrected 2026-07-28.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_nltk_download_false_is_not_logged_as_success` passes whether or not `nltk` is installed
- [x] #2 The test still fails if the production warning in `_ensure_nltk()` is removed, i.e. the fix does not make it vacuous
- [x] #3 The orphaned, over-indented comment block left at `Chunk_Lib.py:267-268` by the task-842 refactor is removed
- [x] #4 No production behaviour changes: `Chunk_Lib`'s logging and readiness semantics are untouched
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inject a stub `nltk` / `nltk.tokenize` pair into `sys.modules` for the duration of the test so `_ensure_nltk()`'s import succeeds regardless of environment. The existing stubs on `_probe_sent_tokenize` and `_download_nltk_tokenizer_corpora` already prevent the fake tokenizer from being called, so the stub only has to satisfy the import.
2. Verify the test passes both with `nltk` installed and with it genuinely uninstalled. (Deviation from the first draft of this plan, which proposed a `sys.meta_path` blocker: actually removing the package tests the real condition rather than a simulation of it, and the install/uninstall is cheap.)
3. Mutation-check the guard per `lessons-testing-evidence.md`: temporarily delete the production warning and confirm the test fails.
4. Delete the dead comment block at `Chunk_Lib.py:267-268`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The production warning was never missing. The task's original premise was wrong and is corrected in the Description above; what shipped is a test fix plus a dead-comment cleanup.

**Approach.** `test_nltk_download_false_is_not_logged_as_success` now injects a stub `nltk`/`nltk.tokenize` pair into `sys.modules` via `monkeypatch.setitem`, so `_ensure_nltk()`'s real `import nltk` succeeds regardless of which extras are installed. The stub only satisfies the import: `_probe_sent_tokenize` is already stubbed to return False, so the fake tokenizer is never called. Setting `NLTK_AVAILABLE = True` alone was never sufficient, and that gap is now called out in a comment at the test so it is not reintroduced.

**Verified.**
- Passes with `nltk` installed, and with it uninstalled (both run explicitly, not inferred).
- Mutation-checked per `lessons-testing-evidence.md`: deleting the `logger.warning` in `_ensure_nltk()` makes the test fail; restoring it makes it pass. The guard is not vacuous.
- `Tests/Utils/test_startup_polish_regressions.py` + `Tests/Chunking`: 106 passed, 1 skipped.

**Trade-off.** A stub was chosen over `pytest.importorskip("nltk")`. Skipping would also stop the misleading failure, but it would silently drop coverage on exactly the minimal installs where the fallback path matters most. The logic under test is ours, not nltk's, so it should not require the package.

**Production change is comment-only:** the orphaned, over-indented two-line comment at `Chunk_Lib.py:267-268` — dead text left when task-842 replaced the download block — was removed. No logging or readiness semantics were touched.

**Modified files.**
- `Tests/Utils/test_startup_polish_regressions.py` — stub injection, `types` import, explanatory comment
- `tldw_chatbook/Chunking/Chunk_Lib.py` — dead comment removed
- `backlog/docs/lessons-testing-evidence.md` — new entry "A missing extra fakes a code regression", recording the wrong root-cause this produced
<!-- SECTION:NOTES:END -->
