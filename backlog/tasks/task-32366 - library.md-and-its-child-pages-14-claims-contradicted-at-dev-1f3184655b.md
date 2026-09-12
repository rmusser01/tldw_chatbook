---
id: TASK-32366
title: 'library.md and its child pages: 14 claims contradicted at dev 1f3184655b'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-11 06:20'
labels:
  - library
  - docs
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Contradictions found by the critique-10 assessors: the landing footer's 'n new note' (live ctrl+n); Search/RAG's 'enter select evidence' hint (absent); a Search⇄RAG mode toggle (absent); Chunking Lab 'above' its gloss (below); the Notes editor's 'ctrl+s save note' chip (absent); the rail beside Notes at ≥120 columns (collapsed at 235); the 'Nav' handle label (renders '--->'); 'Back to Get started never offered after graduation'; the bulk-delete banner copy; the 'cancel delete' footer string; a dirty-edit veto on Escape (notes autosave instead); the Find bar behaviour; glosses 'shown consistently' (4 of 7 rows); empty Browse rows without pager mechanics (Skills/Collections/Trash keep them). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each claim is either made true by the guide or by the surface, with the task that fixed the surface named in the stamp
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read library.md, library/notes.md and library/search-and-rag.md at the wave's dev tip, after Tasks 1-7 landed.
2. Drive the seeded and a starter profile live at 235x52 and capture one screen per claim.
3. Keep only claims a capture supports; rewrite the rest to the shipped behaviour; delete a claim no capture can reach.
4. Append the reconciliation stamp to each page's stamp block.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Docs-only. All 14 rows were re-checked live at 235x52 (socket `crit10-docs`, seeded
profile plus a fresh starter profile built with `mkprofile.sh`); captures are under
the session scratchpad at `crit10/wave/docs/caps/task8/`.

Six rows were already true on dev and only needed verifying:

| # | Claim | Closed by | Live evidence |
|---|---|---|---|
| 2 | Search/RAG `enter select evidence` | task-32346 (#2602) | footer reads `enter run search` with the caret in the query box |
| 6 | rail beside Notes at >=120 columns | task-32355 (#2603) | rail stands beside the Notes list at 235 |
| 7 | the `Nav` handle label | task-32355 (#2603) | `N`/`a`/`v` painted down the collapsed handle |
| 8 | "Back to Get started" after graduation | task-32349 (#2598) | the row is present after Explore, gone once the first note exists |
| 12 | the Find bar behaviour | task-32348 (#2602) | media Reader footer carries `ctrl+f find` |
| 14 | empty Browse rows without pager mechanics | task-32354 (#2599) | single-page Skills shows `1-2 of 2` only |

Seven rows were corrected in `Docs/User_Guide/library.md`:

- #1 the landing footer is `ctrl+n new note`, not a bare `n`.
- #4 the Chunking Lab buttons render **below** their gloss, not above.
- #5 the Notes editor advertises `esc back to notes` and no save key.
- #9 the bulk-delete banner is quoted from the live string ("Delete 2 selected
  items? You can undo right away, or restore later from Trash.").
- #10 the armed-delete footer chip is `esc cancel delete`.
- #11 the bullet now distinguishes the three editors instead of generalising.
  Notes autosave, so Escape leaves at once (DB-proven live: the row's age went to
  "now"); only a blocked save holds a note. Prompts and Skills do NOT autosave and
  DO veto Escape while dirty (`library_prompts_controller.py:3456-3459`,
  `library_screen.py:24802-24810`) — Skills notifies, Prompts refuses silently,
  which is what "Escape does nothing" looks like from the outside and is filed as
  task-32393. `library/prompts.md` and this bullet now state the veto in the same
  words.
- #13 softened rather than "fixed at the source". The brief's alternative (add
  glosses for Conversations and Notes, rename Collections' to "saved pages") is
  the exact reverse of a standing pin: `Tests/Library/test_library_shell_state.py`
  `test_jargon_rows_carry_plain_language_subtitles` asserts
  `browse-conversations == ""` and `browse-notes == ""` ("already-plain rows stay
  unglossed so the rail doesn't stutter", F-013) and pins Collections to
  "saved captures". `library_shell_state.py` is therefore unchanged. The page now
  names which rows carry a gloss, says Conversations and Notes are bare by design,
  and states the measured reason Collections reads bare: its gloss needs 34 cells
  of row width (`LibraryRail._row_label` renders it from 34 up, drops it at 33) and
  the rail's default gives 33 — `project_default_library_width` is a bounded
  fraction (29 cells at 64-120 columns, 35 at 160, 39 from ~181), not a fixed
  width, and a user-set width reaches 48. It does render on the single-pane stage
  below 64 columns, where the rail takes the whole terminal (assessor B's caps
  56/57). The same sentence's short-label list was corrected from "Sets" to
  "Captures" (`short_title="Captures"`).

One row was **refuted, not fixed**: #3 claimed no `mode: ✓ Search ⇄ RAG Answer`
control exists on the Search / RAG canvas. It does — it is the canvas's first
control, composed unconditionally in `library_search_rag_panel.py:407`. The guide's
two example strings and `library/search-and-rag.md`'s four references were kept
unchanged, and search-and-rag.md's stamp records the re-check.

Modified files: `Docs/User_Guide/library.md`, `Docs/User_Guide/library/notes.md`
(stamp only — its autosave story was already correct),
`Docs/User_Guide/library/search-and-rag.md` (stamp only).
<!-- SECTION:NOTES:END -->
