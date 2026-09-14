---
id: TASK-32461
title: 'Library Prompts: three more dirty vetoes refuse without a word'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-12 00:10'
updated_date: '2026-09-14 15:28'
labels:
  - library
  - prompts
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32393 fixed the Escape/Back seam: a dirty Prompts editor now says "Unsaved Prompt changes — Save or Discard changes first." instead of refusing in silence. Three sibling vetoes read the same flag and still say nothing, so the same click-does-nothing-says-nothing defect survives on three other surfaces.

All three refuse on `_flush_library_prompt_save()` returning False (which is exactly `not _prompts_state.dirty`) and return without notifying: the prompt-row switch (`library_screen.py` ~25675, pressing another prompt row while the open one is dirty), select-mode entry (`library_prompts_controller.py` ~1433, pressing "Select" while dirty), and the entry-reconcile path (~33622, a deep link into a prompt arriving while the editor is dirty). The rail-row switch and the app-level navigation guard already notify (`library_screen.py:21405`, `:10485`), so the copy and the pattern exist — these three were simply never wired to them.

The background reconcile path is the one that needs a judgement rather than a copy-paste: a toast raised by something the user did not just press may be noise, so decide whether it explains, defers, or stays silent by design and record which.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing another prompt row while the open prompt is dirty states why the switch was refused, on the same line as the next step
- [x] #2 Pressing Select while the open prompt is dirty states why it was refused
- [x] #3 The entry-reconcile veto's behaviour is decided and recorded in the task (explain, defer, or deliberately silent), and matches what ships
- [x] #4 Each wired refusal is covered by a test that fails if it becomes a silent no-op again
- [x] #5 Navigating into Library while a dirty prompt editor is open states why the move was refused
<!-- AC:END -->

<!-- AC#5 added in fix round 1: the task review found a FOURTH silent sibling the filed
three missed (`library_inspection_admission.py:230`, the navigation-into-Library barrier),
and it is the one route that actually fires today. -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the three filed line anchors against the current files (dev moved).
2. Wire the prompt-row switch and Select-mode entry to the existing _notify_prompt_dirty_veto().
3. Entry reconcile: per the controller's ruling, explain and name the blocked target (no queue).
4. Red-first pins for all three, mutation-tested by dropping each notify.
5. Docs: Docs/User_Guide/library/prompts.md + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All three sibling vetoes now speak; none of them changed what they refuse.

- **Prompt-row switch** (`library_screen.py`, `handle_library_prompt_row`) and
  **Select** (`library_prompts_controller.py`,
  `handle_library_prompts_select`) call the existing
  `_notify_prompt_dirty_veto()` before returning — the same
  `LIBRARY_PROMPT_DIRTY_VETO_COPY` Back/Escape, the rail-row switch and the
  app-level navigation guard already raise. No third variant.
- **Deep link** (`library_screen.py`, `_open_library_item_by_id`'s `"prompt"`
  branch) raises a target-naming variant instead: the notifier grew one
  optional `blocked_target` kwarg and formats the new
  `LIBRARY_PROMPT_ENTRY_DIRTY_VETO_COPY` when it is set. Nothing is queued —
  see `## Decision`. The target is `Prompt <id>`, the editor's own
  unresolved-name shape, because no caller passes a display name to that seam.

Line anchors: all three had moved since filing (25675→25847, 1433→1439,
33622→33803) and were re-verified before editing.

Deviation from the brief's "touch `library_screen.py` at exactly one place":
the entry-reconcile seam lives in that file too (the filed `~33622` anchor is
`library_screen.py`, not the controller), so it is three one-line-ish,
non-structural touches there — the veto seam, the deep-link seam, and the
mechanical delegator whose signature had to carry the new kwarg.

Evidence: `Tests/UI/test_library_prompt_dirty_vetoes.py` (3 new pins) went
3 red → 3 green, and each pin was mutation-tested by deleting its own notify
(1 failed, 2 passed, each time the matching one). Live at 235x52 on a scratch
profile: the row-switch and Select refusals both raise the toast (captures
`01-row-switch-veto.txt`, `02-select-veto.txt`). The deep-link refusal is not
reachable by hand — every route to a prompt deep link is itself vetoed while
the editor is dirty — so it is covered by the mounted-screen pin only.

Modified: `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Library_Modules/library_prompts_controller.py`,
`tldw_chatbook/UI/Library_Modules/screen_constants.py`,
`Tests/UI/test_library_prompt_dirty_vetoes.py` (new),
`Docs/User_Guide/library/prompts.md`.

### Fix round 1 (task review, 2026-09-14)

**AC#5 — the fourth sibling.** `_flush_library_navigation_sources`
(`library_inspection_admission.py:230`) is the barrier every navigation INTO
Library crosses, and it refused a dirty prompt in silence five lines above a
skill veto that speaks. It also returns before the deep-link seam, so the AC#3
line shipped in round 0 was correct but dormant while the live defect stood.
One `self._notify_prompt_dirty_veto()`, in that twin's own shape (the
`is_current()` check split out, so the flag branch reads like the skill one).
Red-first (`runs/f1-red.txt`: "the route into Library refused and said nothing
at all"), green (`runs/f1-green2.txt`), mutation-tested by deleting the call
(`runs/f1-mut.txt`: 1 failed, 3 passed, the right one). Live at 235x52:
`captures/03-route-into-library-veto.txt` — palette ▸ "Media & Content: Open
Media Library" while the Prompts editor is dirty now raises the toast, the
route does not happen, the editor keeps its "Unsaved changes".

**Caller sweep of `_flush_library_prompt_save` — 8 sites, 8 now speak.**
This is the check that closes the class rather than the instances; it is what
would have caught the fourth site before review.

| # | Call site | Gesture | Disposition |
|---|---|---|---|
| 1 | `library_inspection_admission.py:230` | any route INTO Library (palette, Console hand-off, legacy alias) | **speaks** — new in this round; live-captured |
| 2 | `library_prompts_controller.py:1440` | **Select** | speaks (round 0); live-captured |
| 3 | `library_prompts_controller.py:1592` | **Import…** | speaks — predates this task |
| 4 | `library_prompts_controller.py:3491` | Back / Escape (`_exit_library_prompt_editor_guarded`) | speaks — task-32393 |
| 5 | `library_screen.py:10634` | app-level navigation guard (`flush_pending_work`, tab switch) | speaks — predates this task |
| 6 | `library_screen.py:21580` | rail-row switch | speaks — predates this task |
| 7 | `library_screen.py:25851` | prompt-row switch | speaks (round 0); live-captured |
| 8 | `library_screen.py:33829` | deep link (`_open_library_item_by_id`) | speaks (round 0); pinned on the mounted screen — no hand route reaches it while dirty, because sites 1, 5 and 6 block every approach |

No ninth site: the grep is `_flush_library_prompt_save()` across
`tldw_chatbook/`, and the two remaining hits are the definition and the
controller's late-binding accessor.

**Deep-link copy, id → title (review F-5).** Shipped: `_open_library_item_by_id`
takes an optional `display_name`, and both hand-reachable callers pass the value
they already hold — the evidence card's `row.title`
(`library_rag_search_controller.py`) and the hub recent row's title (carried on
the button beside the `(source_type, record_id)` pair it already routes on).
No lookup, no fetch. Empty falls back to `Prompt <id>`, and both branches are
pinned.

**Review F-8.** The id parse moved above the veto, so a malformed deep link is
dropped without naming its garbage back; pinned (`"not-an-id"` raises nothing).

<!-- SECTION:NOTES:END -->

## Decision

**AC#3 — the entry-reconcile veto explains** (the controller's call, recorded
verbatim; revisitable):

> RULING on AC#3 (mine, so you do not stall): the entry-reconcile veto
> **explains**. A deep link into a prompt that silently evaporates is the same
> click-does-nothing defect one layer further out — the user pressed something
> somewhere to cause it. Name the blocked target and the way out in one line
> (e.g. the pending prompt's title plus "save or discard the open prompt
> first"). Do NOT build a queue that applies the deep link after the save —
> that is machinery for a case nobody has asked for.

What ships matches: `_open_library_item_by_id`'s prompt branch raises
`LIBRARY_PROMPT_ENTRY_DIRTY_VETO_COPY` — "Can't open Prompt 7 — Save or
Discard the open Prompt first." — and still returns `None`, so the link is
dropped exactly as before and nothing is queued. The target is named by id
rather than title because no caller passes a display name down to that seam
(the editor's own unresolved-name fallback has the same shape).
