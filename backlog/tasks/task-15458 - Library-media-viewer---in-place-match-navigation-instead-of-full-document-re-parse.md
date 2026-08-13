---
id: TASK-15458
title: Library media viewer: in-place match navigation instead of full-document re-parse
status: In Progress
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit — plausibly the single worst click in the app: the media viewer holds the entire document as one Markdown widget (`Widgets/Library/library_media_viewer.py:148`), and next/prev match (`library_screen.py:23476`), mode toggle (`:23431`), and search submit (`:23362`) each perform a full-document markdown re-parse plus a whole-screen remount — multi-second per click on a long transcript on constrained hardware. Separately, the media panel's content search re-parses the document twice per keystroke via `Markdown.update("")` followed by `Markdown.update(content)` (`Widgets/Media/media_viewer_panel.py:1000-1025`, reached from `:1223/:1266`), with no debounce.

Fix direction: keep the Markdown widget mounted and move match-highlight and scroll-to-match to in-place updates; drop the empty pre-update; debounce the search box. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Match navigation does not re-parse the document or remount the screen (evidence)
- [ ] #2 Search-while-typing performs at most one deferred re-render per debounce window and never the double update(\"\")/update(content) parse
- [ ] #3 Match highlighting and scroll behavior preserved (tests); click latency before/after on a long document recorded
- [ ] #4 Literal async pytest commands run on Windows without mutating the guarded network families, while ordinary same-thread and concurrent-thread application egress remains blocked and recorded
<!-- AC:END -->

## Implementation Plan

1. Add mounted tests and a focused `LibraryMediaContentSearchControls` widget whose match-only updates preserve navigation identity and focus.
2. Add mounted tests and a lazy `LibraryMediaContentBody` that stores Raw search state, mounts each selected mode at most once, and applies latest-request-wins visibility.
3. Integrate the focused children through `LibraryMediaViewer` and replace Library screen recomposes with narrow query, match-index, mode, and post-layout scroll synchronization.
4. Add mounted legacy-panel tests and a generation-guarded 250 ms content-search debounce; remove the empty Markdown cache-busting update.
5. Record deterministic before/after latency and parse/identity evidence, run focused and full verification, complete rendered keyboard UAT, and close the task documentation.

Detailed plan: `Docs/superpowers/plans/2026-08-12-library-media-viewer-inplace.md`

ADR required: yes

ADR path: `backlog/decisions/058-thread-scoped-test-socketpair-exemption.md`

Reason: the media-viewer changes apply existing Textual ownership patterns, but
the review-expanded verification scope changes the repository-wide test network
security boundary and its runtime interception contract.

## Implementation Notes

### Task 3 pre-change product-path evidence

The exact Task 3 focused command selected 16 tests. Before production changes,
five new regressions failed for the intended missing behavior: the viewer was
replaced on search/navigation, the persistent content body was not mounted,
inactive Raw/Rendered children were removed, and Rendered search state did not
reach a later Raw mount. The Enter-only regression already passed.

The deterministic 2,000-line Markdown document contained exactly 100 matching
source lines. The isolated repeat of the identical submit/Next/Prev sequence
recorded `TASK-15458 latency median_ms=1091.689`. The screen stayed at object ID
`2237538817072`, while the viewer IDs were `2235866381104`, `2237674891232`,
`2235874008944`, and `2235896514272`, and the Markdown IDs were `2235871484416`,
`2235878885376`, `2235889419760`, and `2235894721824`. The construction proxy
therefore observed four distinct viewers and Markdown objects, plus four
parse-triggering `Markdown.update` calls (initial mount plus one for every action).
These process-local IDs are evidence of replacement, not stable identifiers
expected to recur across runs. The preceding full-slice run measured 1239.726 ms
and the same four-object/four-parse shape.

ADR required: no additional ADR for Task 3. This slice applies the already
reviewed widget-ownership interface without changing storage, runtime, security,
dependency, or cross-module service boundaries. The parent task's ADR-058 remains
the governing decision for the separate Windows network-guard verification scope.

### Task 3 implementation and post-change evidence

Integrated the reviewed search-controls and persistent-body components through
`LibraryMediaViewer`, then replaced Library screen query, navigation, and mode
recomposes with narrow mounted-viewer synchronization. Search remains
Enter-submitted; navigation wraps and schedules source-line scrolling after a
refresh boundary; Rendered remains the Markdown default; Raw and Rendered remain
mounted after first use; and async mode synchronization delegates to the existing
latest-request-wins body contract. No dependencies, persistence, workers, legacy
Media debounce, or geometry rules changed.

The required focused product-path command passed all 16 selected tests. Its
2,000-line/100-match evidence recorded `TASK-15458 latency median_ms=120.913`,
one stable screen ID (`2247968276448`), one stable viewer ID
(`2247963750576` across all four observations), one stable Markdown ID
(`2247965990096` across all four observations), and one `Markdown.update` call
for the initial mount only. An isolated repeat measured 89.561 ms with the same
one-viewer/one-Markdown/one-parse shape. This compares with the isolated
pre-change 1091.689 ms median, four viewer objects, four Markdown objects, and
four parses.

The full `Tests/Library/test_library_media_content.py` suite passed 9 tests. A
mutation that restored `self.refresh(recompose=True)` in match navigation made
the identity/focus/parse regression fail at the viewer identity assertion; after
restoring the narrow path, that regression passed. Task-scoped Ruff checks passed.
The whole-file Ruff baseline still reports pre-existing `E721` findings in
`library_screen.py` and one pre-existing unused import in `test_library_shell.py`;
the scoped verification excluded only those existing rule findings. `git diff
--check` was clean.
