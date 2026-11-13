---
id: TASK-40327
title: "Opening Find in the Library Media viewer re-parses the whole document"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 05:13'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- media
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_media_controller.py
- tldw_chatbook/Widgets/Library/library_media_viewer.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (lib stream; TASK-39152 recorded the extra parse as debt and pinned it). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Opening Find on a large document pays a full Markdown re-parse (the test's 49 KB case) for a toggle that only adds a bar, and it recomposes every viewer child, so focus has to be rescued afterwards. Introduced by 7b04023420.

**Path.** `_toggle_library_media_find` (`UI/Screens/library_screen.py:35041-35068`) sets `find_open=True` and syncs the viewer. In the media controller's sync, `viewer.find_open != state` fails the unchanged check (`library_media_controller.py:4463`), so it assigns `viewer.find_open` (`:4510`) and calls `viewer.refresh(recompose=True)` (`:4561`). `LibraryMediaViewer.compose` rebuilds `LibraryMediaContentBody(content=...)` (`Widgets/Library/library_media_viewer.py:755-767`), whose `Markdown.update` re-parses the whole document. `Tests/UI/test_library_shell.py:11886-11889` pins exactly 2 parses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening and closing Find causes no parse of the document after the first; the latency proxy test is tightened from 2 parses to 1
- [x] #2 The content body widget keeps its identity across a Find toggle, and focus stays where it was
<!-- AC:END -->

## Implementation Plan

1. Re-check the path at base 26b2e677dd: the large-document proxy still pins 2 parses; probe the Find toggle's UI-loop CPU (loop `thread_time`) on the 2,000-line document.
2. Patch the Find bar in place: a `LibraryMediaViewer` method mounts the search controls before the content body (removing them, after clearing the query from the body in place, on close), and the media controller's viewer sync calls it as the last term of its unchanged check instead of comparing the Find fields.
3. Keep the recompose for what only a recompose can do (a new query, no body, an Analysis Raw/Rendered swap).
4. Tighten the proxy to 1 parse; add a test for body/Markdown identity, parse count and focus across open/close; re-measure.

## Implementation Notes

- **Premise held at base 26b2e677dd** (2026-11-13): the viewer sync's unchanged check still compared `find_open` and the focus token, so opening or closing Find recomposed the whole `LibraryMediaViewer` and rebuilt `LibraryMediaContentBody`; the large-document proxy pinned 2 parses.
- **Fix.** `LibraryMediaViewer.sync_find_bar` mounts the search controls directly before the content body (where both the Read and the Analysis bodies compose them) or removes them; on close it first applies the cleared query to the body in place (`sync_search`). It returns True when nothing changed, False only for what still needs a recompose: no content body, a new query (`sync_query_state` owns submits) or the Analysis Raw/Rendered swap a query flip forces (now one helper, `_query_flip_swaps_analysis_mode`, shared with `sync_query_state`). The media controller's unchanged check calls it as its last term instead of comparing `find_open`/`content_query`/`content_match_index`/token, and consumes the token on the unchanged path; the opening bar takes focus through its existing `focus_on_mount` token.
- **Tests.** `test_library_shell_media_viewer_inplace_large_document_latency_and_parse_proxy` tightened from 2 parses to 1 (red at base: `assert 2 == 1`). New `test_find_toggle_keeps_body_parse_and_focus` (`Tests/UI/test_library_media_reader_match_nav_t22209.py`, census node): open, submit, close with the button (focus stays on Find), reopen (focus in the input), Escape (focus on Find) -- content body, Markdown and Find button keep their identity and the 2,000-line document is parsed once (red at base on the body identity). `test_opening_find_costs_no_extra_focus_move` now pins one focus move: the old first move parked focus on the items grip during the Reader recompose (red at base). Two comments that said opening Find recomposes the viewer were corrected.
- **Measured** (probe: Find opened and closed 6 times per launch on the 2,000-line document, UI-loop `time.thread_time()` from press to settled, `gc.collect()` first; base and head back to back, 2 launches each, load 38-41): base open 208-240 ms and close 204-270 ms, one parse each; head open 79-118 ms and close 87-112 ms, no parse.
- **Covering runs** (`-n 4`, load 50-70): the Find/viewer files (match-nav, crit10 viewer, render fixes, reader flow, side-by-side, reader shell, Library media content/state); every non-RecoveryRequired failure was compared against base with the same command: identical apart from load flakes that fail on either side (`NoMatches` for a Reader button before the Reader mounted). Plain files hitting the local RecoveryRequired trap were run under a temporary module-level `bootstrap_profile` marker, then the marker was removed.
- **Lane.** The new node is in `scripts/ui_pr_gate_census.txt` (tail): 13.4 s call, 24.4 s total serial with the UI lane's flags at load 42-46.
- Files: `tldw_chatbook/Widgets/Library/library_media_viewer.py`, `tldw_chatbook/UI/Library_Modules/library_media_controller.py` (5267 -> 5273, inside its pin), `Tests/UI/test_library_shell.py`, `Tests/UI/test_library_media_reader_match_nav_t22209.py`, `Tests/UI/test_library_media_reader_flow.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

