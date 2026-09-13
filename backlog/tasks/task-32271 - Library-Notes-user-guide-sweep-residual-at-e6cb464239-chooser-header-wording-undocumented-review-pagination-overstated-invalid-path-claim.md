---
id: TASK-32271
title: >-
  Library Notes user guide sweep residual at e6cb464239: chooser header
  wording, undocumented review pagination, overstated invalid-path claim
status: Done
assignee: []
created_date: '2026-09-10 18:05'
updated_date: '2026-09-13 03:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The task-32141 guide sweep landed and the evidence assessor's 38-row conformance table is mostly VERIFIED -- Import once's exclusivity, the Obsidian skip rules, Info Properties, the delete copy and receipt shape are all verbatim-accurate. Three claims are still off, and none of them belongs to a code task in this batch:

- the guide says the chooser header "reads **Add from files**" until you choose; live it reads "Add files to Library notes.";
- the review's pagination (`Page 1 of 3` with per-page group counts) is entirely undocumented;
- "An invalid path shows an inline reason and leaves the dialog open" is overstated while the reason is painted into the dialog border (the code side of that is task-32251; this is the doc stamp).

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Merge origin/dev (7159fc0b99, every wave-3 code group) into the docs branch.
2. Enumerate every guide claim the wave touched: `git log 4a14b3f36f..origin/dev` on the five guides plus the "Verified against" stamps for 32143, 32146, 32186, 32242-32272, 32294-32298, 32389/32390, 32467.
3. Walk each claim live on dev at 235x52 and 100x30 (60x24 for the below-64-column claim) on a seeded scratch profile with a git-backed vault under `$HOME/.cache/tldw-crit/t12/`; one capture per claim under `wave3-caps/docs-sweep/`.
4. Fix every contradiction with the supersession treatment; add the undocumented pager; consolidate duplicate stamps; refresh the stamp with the commit walked.
5. Riders for defects found; renumber the colliding task-32472; run Tests/Docs and the backlog-id uniqueness test; inventory + bundle checks; PR.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each claim above matches the live surface, or is corrected in the guide
- [x] #2 The 'Verified against' stamp on `Docs/User_Guide/library/notes.md` is refreshed with the commit it was checked at
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Walked every wave-3 guide claim live on dev 7159fc0b99 (merged into
fix/library-notes-wave3-docs) at 235x52, the compact claims at 100x30 and
the below-64-column claim at 60x24, on two seeded scratch profiles (t12,
t12b) with git-backed Obsidian vaults under `$HOME/.cache/tldw-crit/t12/`.
Captures: `wave3-caps/docs-sweep/` (01–13 editor and list; 20–42 Import
once, review, receipt, backlinks, delete/undo; 43–67 lasting sync; 70–84
Folder files, Session Git, export; 90–99 Console capture-as-note and rails;
100 ingest browse; 101–107 at 100x30; 110–111 at 60x24; b01–b21 the second
profile: paged review, seeded activation, pause/resume, Save as… ▸ Note).

The three residuals this task named, plus what the walk found:

- Chooser heading — live "Add files to Library notes." / "Choose how files
  should relate to Library notes." (cap 20); guide superseded ("Add from
  files" was the toolbar button, never the heading).
- Review pagination — a 67-source vault with a collapsed 45-file run fits
  one page (cap 26), so the pager was walked on an 81-source vault of
  single-note folders: three stacked lines under the page's last group —
  "Previous page unavailable — this is the first page" / "Page 1 of 4" /
  "Next page", "Next page unavailable — this is the last page" on page 4,
  "New (25 of 73 on this page)" headers (caps b02–b04). Documented.
- Invalid path — the reason renders on a row under the field ("The file
  must exist", cap 22), as task-32251 shipped; the guide's overstatement was
  already corrected by that group, and the picker paragraph was wrong in a
  different way: Import once and Keep a folder synced open the
  files-or-one-folder dialog whose field is "File name" and starts empty
  (caps 21–24, b05a); only Folder files' "Choose File Notes Folder" has the
  pre-filled "Folder path" (caps 71–72). Superseded.
- Session Git — the dialog is "Trust repository for session changes?" (cap
  77), not "Trust Session Git repository?"; superseded in file-notes.md.
- Stamps consolidated: the two capture-console stamps (chat-basics.md and
  notes.md) and the two import-review stamps each folded into one.
- Everything else held: third-part row disambiguator (01), one selection
  count (02–03), Tab/Ctrl+End/Shift+Tab/delete prompt (05–12, 39), Preview
  focus (09), chrome strip at 235x52 and 100x30 and absent at 60x24 (04,
  104, 111), Linked from through import/trash/undo (31–38), `.git` skip and
  collision default (26, 41–42), receipt denominators (29), compact select
  strip and ‹ Library / Notes cue (103–107), below-64 stage (110–111),
  Session Git keyboard/commit (75–82), export refusal (83–84), ingest Browse
  at `[notes] sync_directory` (100), Capture as note → Open note (90–99),
  Save as… ▸ Note titled after the conversation with `console` only (b21,
  read from the database).

Two defects found, reproduced twice each, filed rather than fixed:
task-32518 (activating a lasting-sync root on a profile that already holds
notes leaves the Notes list stale until restart — database 10 → 70, list
stays at 10 through check, rail re-select and source round trip; caps
b06–b11, 48–54) and task-32519 (Resume after Pause always lands in
✕ Failed and leaves the root paused — cause proven with an import-time probe:
pause cascades bindings to 'paused', `observe_root` refuses them with
`binding_review_required` before Resume can re-activate; caps b12–b15,
56–67, `resume-trace.log`). Both are named as known gaps in the guide with
the supersession treatment. Riders 32516/32517 record the two test races the
landings reported; 32520 records the merge-conflict markers found on dev in
`Docs/User_Guide/console/agent-runs-and-tools.md` by the landing sweep.
Rider 32472 renumbered to 32515 (dev's 32472 is older).

Tests: Tests/Docs (22 passed; the two README.md nodes fail identically on
the dev baseline — dev-side), Tests/CI/test_backlog_task_id_uniqueness.py,
and the guide-reading UI files (w3_layout, crit10_layout, crit8_polish_shell,
files_sync_journey) compared against the detached dev baseline. Inventory
(603 owners) and bundle checks clean on the merged tree.

Lesson recorded (lessons-live-verification.md): appending a table to a
scratch config the app has already expanded corrupts it, and the app then
opens the real profile — the previous implementer's Console leg did exactly
that for two boots on 2026-09-12 17:42 PT.
<!-- SECTION:NOTES:END -->
