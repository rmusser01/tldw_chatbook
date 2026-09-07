---
id: TASK-31635
title: 'Library media - critique #5 polish batch'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 06:19'
updated_date: '2026-09-07 07:43'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #5 minor observations, grouped: Find counter reads Match 1 of 1 matches; the permanent-delete confirm prints a raw UTC microsecond timestamp; Restore and Delete permanently render fused with identical weight and no keys; ○ Delete shifts one cell when it enables; Find Prev/Next stay enabled at No matches while the pager's disable; Export and Trash stay enabled over a failed empty list; the Sets tab disappears on zero results; a single result auto-loads silently; ] marks an item reviewed as a side effect of moving; the review banner readout goes stale after a click; the Reader keeps showing a live item over Trash; the failed-load placeholder says Select a media item; Rendered/Raw is silently absent for article and document; the rendered H1 is centred; the Import path field and the metadata fields use two field idioms; Open manager is indented one cell further; the review-set toast overlaps the Reader border at 100x30.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed item is fixed or explicitly declined with a one-line reason in this task
- [x] #2 Painted or unit tests pin the fixed items
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Merge dev with PRs E–I first; triage the seventeen items for duplicates against what those PRs shipped (declines need a named trade-off).
2. Task 1 — Reader and Find (items 1, 5, 12, 13, 14). Task 2 — Trash and select-mode chrome (items 2, 3, 4, 6). Task 3 — list, review-set and Import residue (items 7–11, 15–17) plus four Qodo items (18: try/finally on the viewer-sync seam; 19: analysed refresh after save/restore; 20: keyword-probe opt-in; 21: summary_rows docstring).
3. SDD per task (review + scoped re-review), final whole-branch review + fix round; PR J lands last in the wave.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Critique #5's seventeen minor observations plus four review items from the merged PRs' Qodo threads, each fixed with a pin or declined with a named trade-off. Every cross-task fact flows through one seam: `_library_media_list_unselectable()` feeds the Reader placeholder, the canvas presentation and the Export gate; one `media_trash_age_copy` for the Trash row and its confirm (the state's reference clock); one `_library_media_trash_action_disabled_reason` for tooltips, `check_action` and the footer chips; one `_review_cursor_for_display` for both 'X of M' readouts; `library_disabled_action_label(align=True)` honoured at compose and on the in-place patch path. Trade-offs: `Delete forever` (the long label plus the two-cell gap did not fit the Trash pane's 32-cell floor; the confirm still says permanent); the analysed re-projection reads the SQL projection after the save seam rather than trusting the write, because `save_analysis_version` never commits today (task-31942, PR K); the sibling canvases' 'Export selected' still shift (follow-up); the non-Markdown note covers article/document only (follow-up). Reviews: three task reviews (one fix round each for Tasks 1 and 2 — the failed-load predicate was over-broad both times, corrected to failure AND no retained rows; Trash navigation never gated), one final whole-branch review (blocked only on three guide lines and this record).

## Per-item outcomes (critique #5 items 1–17 + review items 18–21)
| # | Item / outcome | Detail |
|---|---|---|
| 1 | Find counter reads `Match 1 of 3` | FIXED — `_status_text` drops the redundant trailing noun |
| 5 | `Prev`/`Next` `○` + disabled at 0 matches | FIXED — both render through `library_disabled_action_label` and `disabled=True` |
| 12 | Empty-Reader placeholder names a failed list | FIXED — both placeholder sites derive from one helper; the browse settle patches it in place |
| 13 | Non-Markdown `article`/`document` note in the mode-strip slot | FIXED — one `Static` note in the slot the toggle used to vacate; text only, no control |
| 14 | Rendered Markdown H1 aligns left with the body | FIXED — one scoped `content-align: left middle` override of Textual's `MarkdownH1` default |
| 2 | Permanent-delete confirm dates the item the way the row does | FIXED — one relative formatter (`media_trash_age_copy`) for the row and the confirm, the row's reference clock passed through; absolute stamp in a tooltip |
| 3 | Restore / destructive action separated, classed and keyed | FIXED — two-cell gap, danger class, `r` restore / `x` delete as footer chips through the existing shortcut seam, dropped while a confirm is armed; label is `Delete forever` because `Delete permanently` plus the gap did not fit the Trash pane's 32-cell floor (the confirm still says permanent) |
| 4 | Select-mode row buttons hold their column when enabled state flips | FIXED — enabled labels padded to the `○ ` marker width via `library_disabled_action_label(align=True)` on the Media canvas (Conversations/Notes still shift — follow-up) |
| 6 | A failed load gates Export… and Trash | FIXED (narrowed) — `Export…` renders `○ Export…` with the reason only when the list is unselectable (failure and no rows); `Trash` navigation is never gated |
| 7 | **Fixed** | `Sets` composes on the fresh-empty page too (`library_media_canvas.py`). It is navigation, not a result, and it can always open (the picker carries its own empty copy and "Read later"), so it is never disabled. The empty page's ONE-recovery-action budget is preserved: that count is the page body's, and `Sets` lives on the title row. |
| 8 | **Declined + announced** | The auto-load of a filter's first result is intentional and pinned: `Tests/UI/test_library_media_reader_flow.py::test_filter_uses_authoritative_search_and_restores_page_three_anchor` waits on "The first authoritative filter result was not selected in Reader." and on the symmetric restore when the filter clears. Trade-off named: removing it would break a pinned two-way behaviour (filter selects / clear restores). Announced instead — a single-hit filter now paints `1 result · Enter opens` on the list's status line. |
| 9 | **Fixed (ruling: wording only)** | Footer chip `] next in set` → `] next (marks reviewed)`. Behaviour untouched; no review-set state code changed; PR I's row decoration untouched. |
| 10 | **Fixed** | Both "X of M" readouts (Reader banner and footer) now resolve their ordinal through one new read-only seam, `_review_cursor_for_display`, which returns the DISPLAYED item's position when the Reader holds a set item and the persisted cursor otherwise — the same rule `_walk_active_review_set_unguarded` has used since Qodo #2333 and `_active_review_loaded_at_last` since Qodo on #2386. **Not a decline**: on BASE the item-state chip already followed a click but the ordinal did not, so the two halves of the banner contradicted each other. No `set_cursor` write is added. |
| 11 | **Fixed (cheaper option: a line, not a cleared Reader)** | The Reader beside an open Trash list paints `Showing a Media item · not in Trash` in the existing `#library-media-reader-identity` slot (task-31277's grammar), driven by a new `trash_list_open` compose input that is also in the viewer sync's unchanged-compare. Chosen over clearing the Reader because clearing throws away the reading position the user returns to; the identity slot already exists, so no new widget and no new row when it is empty. Deviation from the brief's letter: the line lands one row ABOVE the title (the slot's existing position), not under it. |
| 15 | **Declined** | The Import form does not mix idioms. All four top-level fields carry `.library-ingest-field`'s `border: tall` (`▊▔…▎`); the "thick box" (`┏━┓┃┗━┛`) is `outline: heavy` from `.library-ingest-field:focus`, and the canvas focuses the path field on entry — that is the whole difference. Trade-off named: pinning `▊…▎` on the focused path field re-creates task-3302/MI-05 (focus was colour-only, the form rendered byte-identical in monochrome), and giving the metadata fields the box permanently erases the cue. Pinned painted, both directions. |
| 16 | **Declined — shipped by PR H #2470** | The More row is one `ItemGrid` (`#library-media-reader-more-actions`, 15–16 cell columns), so every action shares the column grammar. Painted at 235×52: `Edit metadata` x=95, `Open manager` x=111, `Move to trash` x=127 — a uniform 16-cell pitch, `Open manager` exactly one column in, not one cell. Pinned at both sizes. |
| 17 | **Fixed (ruling: drop the toast)** | `Reviewing N items.` is now emitted only when `_active_review_set_banner()` is `None` — i.e. when the banner the create is about to open cannot carry the fact. The cap-truncation warning and the displaced-set "Paused …" notice stay (the banner carries neither). |
| 18 | **Fixed** | `_after_library_media_viewer_sync`'s chained callback runs the captured `pending()` in a `finally`; the exception still propagates. |
| 19 | **Fixed (save) + already covered (restore)** | `_save_library_media_analysis` (the single seam both producers persist through — Reader Generate/Save and the bulk Analyze run) now calls `_reproject_library_media_analysis_row`: ONE id-scoped `search_media(library_summary=True, id_allowlist=[id], limit=1)`, i.e. the SAME SQL projection the page uses, re-read for that one row and handed to the new `LibraryMediaBrowseController.note_analysis_state()`. Not on the page path (one human-paced gesture), and the seven-key contract is re-validated on the way in. It asks the projection rather than trusting the write's claim **because live those disagreed** — see task-31942 for the save that does not commit. **Restore half:** `_complete_library_media_mutation` already ends every committed mutation with `controller.request(refresh_scope, …)`, a full page re-request that replaces the placeholder wholesale with the SQL projection — so the restore placeholder understates for the frames between the write and that refresh and never over-claims. Named trade-off: no new code there rather than a second targeted re-fetch racing the one already in flight. |
| 20 | **Fixed** | `match_reasons` is now a keyword-only opt-in on `LocalMediaReadingService.search_media` and `MediaReadingScopeService.search_media`; the browse controller's page fetch is the one caller that passes it. The review-set enumeration loop and the selection-ordering pass issue no probe. |
| 21 | **Fixed** | `Tests/UI/library_media_rows.py::summary_rows` has a Google-style docstring (`Args:` for `n`, `start`, `**overrides`; `Returns:`). |
<!-- SECTION:NOTES:END -->
