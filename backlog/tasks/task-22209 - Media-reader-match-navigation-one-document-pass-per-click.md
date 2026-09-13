---
id: TASK-22209
title: 'Media reader match-navigation: one document pass per click'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 05:57'
labels:
  - performance
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22209).

Pre-existing, still live (the markdown re-parse half was fixed; this is the text half).
Per Prev/Next click: `_advance_library_media_content_match`
(`library_screen.py:33451-33484`) rebuilds the viewer state (full content copy), runs
`find_content_matches` over the whole document, then `sync_match_index -> sync_search ->
build_raw_content_renderable` runs a SECOND full `find_content_matches` plus a Rich `Text`
rebuild with up to 3 appends per line over the entire document
(`Widgets/Library/library_media_content.py:16-53`, `:112-134`). 3-4 O(document) passes per
click; noticeable on multi-MB transcripts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Match navigation performs at most one O(document) scan per click (match list cached keyed on content identity + query; the renderable patches highlight styles rather than rebuilding, or its rebuild is measured and accepted)
- [x] #2 Measured before/after on a multi-MB document
- [x] #3 TASK-21134's layout=False mitigation on the search refresh is preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red-first probe: count find_content_matches calls + highlight-plan builds per Prev/Next click on a mounted Reader (expect 2 scans + 1 full Text rebuild per click today) and pin renderable identity.
2. Widgets/Library/library_media_content.py: extract a RawContentHighlightPlan that owns the ONE document pass (its loop derives the match lines AND the spans together, so the separate find_content_matches call inside the renderable builder disappears); a click rewrites only the two Span entries whose style changed. build_raw_content_renderable stays as a thin wrapper.
3. LibraryMediaContentBody caches the plan keyed on (content object identity, stripped query); sync_search and the compose-time raw widget both route through it. Keep TASK-21134's layout=False.
4. library_screen.py: memoize the screen-side match list keyed on (detail object identity, query) so Prev/Next never rebuilds the viewer state nor rescans; both the submit handler and the match advance route through it.
5. Measure per-click wall before/after on a multi-MB document (micro + mounted).
6. Targeted tests: new t22209 probes + Library media content/viewer-state/reader suites + 22207 traversal probes + library shell; --collect-only sweep; preflight.
7. Mutation-test: drop the query from each cache key (stale-match probes must red) and drop the caches (count probes must red).
8. Teardown/failure walk: search cleared mid-navigation; document swapped while a search is active.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Match navigation no longer re-reads the document. Per Prev/Next click the counted
document passes go 3 -> 0 (4 -> 0 counting the viewer-state content copy that
TASK-22208 memoizes in the same batch), and the handler wall on a 2.5 MB / 24,000-line
transcript goes 52.4 ms -> 7.5 ms (101 KB / 1,000 lines: 2.14 ms -> 0.29 ms), measured
back to back on the same machine.

Two caches, one per layer, each keyed on what actually changes the answer:

* `LibraryScreen._library_media_content_matches()` -- the match list, memoized on
  (detail object identity, query). Both the search submit and every click route
  through it. The detail is only ever replaced wholesale by a settling fetch or
  cleared to None, never mutated in place, so identity is a sound document marker;
  arrow-key traversal swaps the document while a submitted query stays live (a row
  *press* blanks the query, `_select_library_media_reader_row` does not), so the
  detail component is load-bearing. A cleared detail releases the entry rather than
  pinning the previous document's content.
* `RawContentHighlightPlan` (`library_media_content.py`) -- the built Rich `Text`
  plus its spans, held by `LibraryMediaContentBody` and keyed on (content object
  identity, stripped query). The plan's build loop derives the match lines AND the
  spans in the SAME pass, which is what removes the renderable's own
  `find_content_matches` scan; spans are aligned with matches by construction. A
  click rewrites the one or two `Span` entries whose style changed -- no rebuild.

Patch vs rebuild: patching won. Rich exposes `Text.spans` as a reference to its
internal list and the span offsets do not move between clicks (the characters are
identical -- the same fact TASK-21134's `layout=False` rests on), so moving the
active highlight is two tuple assignments. `build_raw_content_renderable` stays as a
one-shot wrapper and is pinned byte-for-byte against the pre-task algorithm by a
parametrized equivalence test (CRLF, unicode, repeated needles, blank lines, no
trailing newline, no-match, empty).

Residual, NOT this task: 7.5 ms of the remaining click is Textual's own
`Static.update` -> `visualize()` -> `Content.from_rich_text` (a whole-document
`str.translate` plus a rich->textual `Style` conversion per span), ~91% of the
handler by cProfile. And the click a user feels is ~1.4-1.5 s on that document
either way, because `Widget._render_content` re-renders ALL 45,000 lines of the
auto-height Static per repaint -- the reader body is not virtualized. Neither is
addressed here; both are visible in the measurements above.

Files: `tldw_chatbook/Widgets/Library/library_media_content.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_media_reader_match_nav_t22209.py` (new, 7 probes),
`Tests/Library/test_library_media_content.py` (+34).

Evidence: red-first 3 passes/click (12 `find_content_matches` + 6 renderable
rebuilds over 6 clicks) -> 0. 144 passed across the media reader/content/viewer-state/
22207-traversal suites; `test_library_shell.py` 664 passed / 64 failed with an
IDENTICAL failure set at base f0e896122, and the 4 `test_library_media_side_by_side.py`
plus 1 `test_library_media_reader_shell.py` reds likewise reproduce at base (all
pre-existing). 59,636 collected, 28 collection errors all missing optional deps.
preflight.sh green. Seven mutations, seven reds: query out of either cache key,
detail identity out of the screen key, each cache removed, content identity out of
the plan key, and the cleared-detail release removed.

AC deviation, deliberate: AC#1 says "keyed on content identity + query". The screen
memo keys on the DETAIL object, not the content string, because the content string is
re-derived (a fresh copy) per `build_library_media_viewer_state` call on this base --
keying on it could never hit. The widget-level plan does key on content identity, and
the matcher has no other semantic inputs (case-insensitive, stripped query, no
whole-word/regex option), so the pair of keys is complete.

Rebase recipe for the TASK-22208/22210 batch (verified with `git merge-tree` against
`perf/batch-b-reader-22208-22210`): exactly two conflicts, both in `library_screen.py`,
both the same shape -- take this branch's `matches = self._library_media_content_matches()`
at the submit handler and the match advance, then inside the new helper swap
`build_library_media_viewer_state(detail)` for 22208's
`self._library_media_viewer_state_cached(detail)`. Nothing else conflicts;
`library_media_content.py` auto-merges. With 22208 in, the first click after a query
change also stops copying the document.
<!-- SECTION:NOTES:END -->
