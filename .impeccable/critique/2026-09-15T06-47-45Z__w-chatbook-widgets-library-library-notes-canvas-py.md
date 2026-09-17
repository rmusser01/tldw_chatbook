---
target: "Library ▸ Notes sub-screen: create / edit / Obsidian import / lasting sync / Folder files, first-timer and power user"
total_score: 25
max_score: 40
na_heuristics:
p0_count: 1
p1_count: 7
timestamp: 2026-09-15T06-47-45Z
slug: w-chatbook-widgets-library-library-notes-canvas-py
---
Method: dual-assessor (A: design-review sub-agent · B: detector/evidence sub-agent, isolated scratch profiles and tmux sockets, no prior-review knowledge; the consolidating parent merged the two reports, traced every claimed cause to code read-only, ran a headless A/B probe on the production sync stack to settle the P0's attribution, and grepped `backlog/tasks` for an open task that already owns each finding before filing anything). Target `tldw_chatbook/Widgets/Library/library_notes_canvas.py` = Library ▸ Notes at origin/dev `77eb2601a6`. Live at 235x52 (both) plus 100x30 and 60x24 (B); one empty profile per assessor (first-timer journey) and one seeded profile (10 notes, 11 media, 6 conversations) with a git-backed 65-source Obsidian vault fixture under `$HOME/.cache`. No LLM provider. Prior snapshot: 21/40 at `5fd502dbac` (2026-09-13); wave 4 (#2676 #2677 #2678 #2679 #2681 #2682 #2683 #2684 #2685 #2686 #2687, tasks 32533–32558) landed in between, and 27 of its 28 tasks are Done.

Deterministic scan: `detect.mjs --json` returned `[]` (exit 0) on the canvas file and on `Widgets/Library/`. **Unscannable, not clean** (the detector reads web markup, not Python or Textual CSS — unchanged since #1). Browser visualization is N-A: this is a pty-drawn Textual UI. Evidence = 45 (A) + 66 (B) tmux captures under `notes-crit4/{A,B}/caps/`, DB reads against the profiles' ChaChaNotes databases, `md5`/`find -newer`/`git status` of the vault before and after every write, `git log` against the fixture repository, both profiles' `tldw_cli_app.log`, and a standalone headless probe of the production notes-sync runtime run against both `77eb2601a6` and `5fd502dbac`.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of System Status | 1 | A `⇄ Both ways` root reports "✓ Up to date · Next: Check changes" while the note edited in Chatbook has never reached the file — no pending state exists anywhere; two disagreeing "Next:" lines in one pane; "Saved" printed twice four rows apart |
| 2 | Match System / Real World | 3 | Sync copy is human now, but the Folder-files empty state offers "Use file_notes" — an internal config key as a button label — and the folder-only sync picker labels its field "File name" |
| 3 | User Control and Freedom | 4 | Nothing. Inline delete prompt with Cancel focused, a named "✓ deleted · title" receipt with Undo focused and its own footer chip, "Recently deleted (1)" behind it, Escape returning from every depth, and both destructive paths gated by a mutation-free review |
| 4 | Consistency and Standards | 1 | Three different folder pickers in one sub-screen — three field labels, two hint wordings, one arriving unfocused, one labelled "File name" while it can only pick a folder — plus four wordings for "go back" |
| 5 | Error Prevention | 3 | The per-row `☐ Skip` / `☑ Create new` are mutually exclusive choices drawn as two checkboxes, which invites ticking both |
| 6 | Recognition Rather Than Recall | 3 | The sync root row carries neither name nor path (task-32451), so two roots would be told apart by status text and order alone |
| 7 | Flexibility and Efficiency | 2 | `/` is printed on the list footer two rows away and focuses the **rail** search from every region but the navigator, or types itself into the query |
| 8 | Aesthetic and Minimalist Design | 2 | The sync review spends ~3 screen rows per file; Info leaves ~25 of 40 rows blank under a ragged button column; the Add-from-files chooser stacks three near-identical headings over a 190x40 empty stage |
| 9 | Error Recovery | 3 | The failure row names its cause but not the next action, and "Manual check finished. Review exact effects." offers no Review |
| 10 | Help and Documentation | 3 | The lasting-sync Obsidian toggle — new in this wave — ships a garbled sentence; `notes.md` carries ~30 inline "(Was … — superseded by task-3xxxx)" clauses inside user-facing prose |
| **Total** | | **25 / 40** | **Fair** (63%) |

## Trend

| # | Heuristic | #1 `c4a7b1911f` | #2 `e6cb464239` | #3 `5fd502dbac` | #4 `77eb2601a6` | Δ | What set #4 |
|---|---|---|---|---|---|---|---|
| 1 | Visibility of system status | 2 | 2 | 2 | **1** | **−1** | "✓ Up to date" over a note edit that is not on disk (P0) |
| 2 | Match system / real world | 2 | 3 | 2 | **3** | **+1** | "Use file_notes" as a button label; "File name" on a folder-only picker |
| 3 | User control and freedom | 1 | 2 | 2 | **4** | **+2** | Nothing — the best heuristic on the screen |
| 4 | Consistency and standards | 2 | 2 | 2 | **1** | **−1** | Three folder pickers; four back-control wordings |
| 5 | Error prevention | 1 | 2 | 2 | **3** | **+1** | Mutually exclusive choices drawn as two checkboxes |
| 6 | Recognition rather than recall | 2 | 2 | 2 | **3** | **+1** | The sync root row carries neither name nor path (32451) |
| 7 | Flexibility and efficiency | 2 | 2 | 3 | **2** | **−1** | The `/` accelerator is region-gated and falls through to the rail |
| 8 | Aesthetic and minimalist design | 1 | 2 | 2 | **2** | 0 | Sync review density; Info's blank rows; the chooser's triple heading |
| 9 | Error recovery | 1 | 1 | 1 | **3** | **+2** | A failure names its category but not its next action |
| 10 | Help and documentation | 2 | 2 | 3 | **3** | 0 | Wave 4's own garbled Obsidian sentence; changelog clauses in user prose |
| | **Total** | **16** | **20** | **21** | **25** | **+4** | |

## Design Specificity Verdict

LLM assessment: **authored for this product — genuinely, in the places that matter most — with a generic editor bolted into the middle of it.** The seams between the three worlds are the best work on the screen: the empty list answers the three-worlds question in one sentence at the moment it is asked; Folder files opens with "Nothing is copied into the Library" and, once a file is open, "5 lines of YAML frontmatter above this body are hidden here and kept exactly as they are on disk"; the import review names Obsidian by name and by rule, reports a `.canvas` as "Obsidian canvas — not a note", points png and pdf at Library ▸ Media, and lets a frontmatter `title:` win over the file name; the sync tree marks `⇄ Sync managed` and every row `⇄ Synced placement`. A generic notes app has none of it.

The inside of each world is unremarkable. The note editor is a title box, a body box, three tabs and a save chip — no backlink affordance in Edit, no folder or keyword context, one Console button. And three folder pickers are plainly three unrelated generic dialogs that happened to land on the same screen. The surface's identity is carried almost entirely by its copy, which is what makes the one place the copy lies (the P0) so expensive.

Deterministic scan: unscannable (above). Visual overlays: none possible.

## Overall Impression

The wave that preceded this critique was the most effective so far: heuristic 9 moved 1 → 3 and heuristic 3 moved 2 → 4, both on work that shipped — honest refusals, a reachable Import once, repeat detection, a local clock, footer chips, receipts. Four points of net movement, the largest single-wave gain in the series.

What holds the score at 25 is that the same surface still contains one state that is not a measurement. Everything else on this screen states its unavailability honestly — "Unavailable - server sync-folder capability not installed", "Retarget/Disconnect unavailable — not in this release; nothing on disk or in Notes changes", "Sort unavailable — clear the filter". The sync root row is the single place that pattern breaks, and it breaks in the direction of false comfort. For a local-first tool whose promise is that the files are the truth, that is the one lie that costs the most.

## Priority Issues

**[P0] A `⇄ Both ways` sync root reports "✓ Up to date" while a note edited in Chatbook has never reached the file.** Editor said "Saved 23:08"; `md5 vault/Daily/2026-09-07.md` unchanged before the edit, 6 s after, 26 s after, and after an explicit **Check changes**; the row read "✓ Up to date · Next: Check changes" throughout; the manual check then set "Manual check finished. Review exact effects." with nothing to review and no Review control; no receipt. Task **32604**. Attribution and cause below.

**[P1] Lasting sync plans a fresh create for every note Import once already made from the same folder** — `Create a Library note` ×54 against Import once's own `Unchanged repeat (54)` minutes earlier; activating yields 108 notes. Task **32605**.

**[P1] "Choose File Notes Folder" opens with no keyboard focus**, so Folder files is mouse-only — a hard blocker for a keyboard-only user. Task **32606**.

**[P1] Inside Info the footer says "enter run action" for every control including Delete**, two stops show no focus at all, and Delete is rendered dimmer than its siblings. Task **32607**.

**[P1] Two of six cells are not completable by keyboard** — Tab runs past "Activate reviewed root" into the rail, and the Session Git commit buttons sit 30 rows below the form. Task **32608**.

**[P1, consolidated to medium] `/` focuses the rail search outside the navigator** while the Notes footer advertises it everywhere (B verified the navigator path works; the accelerator is region-gated). Task **32609**.

**[P1] Every sync root row is titled "Sync folder (name unavailable before cutover)"** — already owned by open task **32451**, not re-filed.

### The P0 — attribution PROVEN, cause PROVEN

**Verdict: independent of wave 4.** A standalone headless probe built on the production runtime, executor and controller (the shape `Tests/UI/test_library_notes_files_sync_journey.py::_start_real_conflict_stack` uses) activates one bidirectional root over a real folder and a real ChaChaNotes DB, edits the bound note the way the editor's save does, waits past the watcher's poll, then runs the controller's `sync_now`. Run against **77eb2601a6** and against a `git archive` extraction of **5fd502dbac**, the results are identical: the row stays `up_to_date` / `Check changes`, the file is unchanged, `sync_now` sets `phase=review` and "Manual check finished. Review exact effects." without touching the row, and `request_sync_now` returns `[('update_file', 'note_changed')]` — which `apply_reviewed` then writes correctly. Wave 4's own pre-fix probe recorded the same asymmetry in task-32534's description. Critique #3 did not see it because its walk also edited the disk, which woke the watcher and carried the Chatbook edit across as a side effect.

**Cause, three links, all read off the code.**
1. Nothing on the note side ever triggers sync. `_ProductionRuntimeAdapter.changed_root_ids` (`Notes/notes_sync_runtime.py:1063-1078`) signs only filesystem metadata — `display_path, device, inode, size, modified_ns, changed_ns` per discovered file (`_discovery_signature`, :1048-1061) — and `PollingNotesSyncWatcher` is the only producer of hints (:1919, :2979; no other `schedule_hint` caller exists anywhere in the tree). A note save bumps the note's version and nothing else.
2. The root row is projected from stored root state, never from a fresh plan, so it keeps "✓ Up to date" over a pending `update_file`.
3. The review the manual Check builds has no door. `LibraryNotesSyncController.sync_now` (`UI/Library_Modules/library_notes_sync_controller.py:1378-1419`) installs it and sets `phase="review"`, but `handle_library_notes_lasting_root_action` (`UI/Library_Modules/library_notes_controller.py:5152-5153`) leaves the view on `lasting_roots` for the `check` action; only `library_notes_add_from_files_canvas.py:482` renders `phase == "review"`, and the roots canvas offers a **Review** button only when `next_action == "review_changes"` (`library_notes_sync_roots_canvas.py:174-176`), which only the automatic pass can set.

The planner is not at fault. It returns exactly what `_plan_bound` specifies (`notes_sync_reconciler.py:610-630`).

## Like-for-like versus 21/40

### Prior P0/P1 → today

| #3 task | Today at `77eb2601a6` |
|---|---|
| 32533 app exits with `InvalidSelectValueError` | **Fixed** (#2678) — zero `unhandled_exception` across four profile walks |
| 32534 sync shows "Up to date" beside "Manual check failed" | **Fixed** (#2681) for the pairing and the receipts; the note→file half is this critique's P0 |
| 32535 sync review lists "Safe item N" with no file names | **Fixed** (#2679) — rows name file, folder and effect; Import once's Obsidian pass runs here too |
| 32536 "Use in Console" fails closed with two messages | **Fixed** (#2677); not re-exercised live (no provider configured) |
| 32537 Preview's footer never names focus | **Fixed** (#2683) — the chip is present in Edit and Preview. Info was not covered (32607) |
| 32540 Import once not completable by keyboard | **Fixed** (#2685) — all of B's K12–K15 pass |
| 32541 re-import re-creates every structured source | **Fixed** (#2676) — a second import reads `Unchanged repeat (54)`, every row defaulted to Skip |
| 32542 autosave clock is UTC | **Fixed** (#2676) — "Saved 22:50" at 22:50 local |
| 32544/32546/32547/32549/32557 layout | **Fixed** (#2684) — no "Remove pl" clip; 60x24 collapses correctly |
| 32539 focus vanishes after delete | **Fixed** (#2683) — Undo focused with "enter undo delete" |
| 32548 duplicates have no id outside the list | **Fixed** (#2683) — `#a7d8` / `#fb48` in list and editor |
| 32550 `/` re-focuses with stale text | **Fixed** (#2683); a different `/` branch is 32609 |
| 32552 Folder-files polish | **Fixed** (#2682) — keys, hidden folders, Git truth |
| 32551 Preview repeats the title | **Partly** — the frontmatter-title case survives (32620) |
| 32538 wrong word count | **Partly** — 5,453 correct in the editor, 5454 in Info (32623) |
| 32553 three names for "go back" | **Partly** — A counts four wordings still (32624) |

### Wave-4 regressions

- **PROVEN.** The lasting-sync Obsidian toggle ships a garbled sentence: *"Turning it off lasts until you quit Chatbook: a vault is offered it again, on, on the next start."* `git show 5fd502dbac:…library_notes_add_from_files_canvas.py` has zero occurrences of the string and no `notes-sync-obsidian` checkbox at all — the whole toggle is new in commit `29cd22159e` (task-32535, #2679), source lines 425-426. Task **32610**.
- INFERRED: "More below — scroll." over 21 blank rows on the same, now larger, pane; a stale "Resolution history unavailable — it starts after this root is activated" left standing after activation; and skip granularity that now disagrees between the two vault paths (folder-level in Import once, file-level in the new sync review).
- **Not** regressions, despite appearances: the P0 (identical pre/post, above) and the `/` region gate (`library_screen.py:24746-24751`, predates the wave).

## Docs vs live

Wave 4's sweep (32558, #2687) re-verified ~300 claims, so each discrepancy is either **new** — a claim the sweep itself added for a wave-4 fix — or **missed** — a claim checked only on its happy path. Four new (Receipts promises "Wrote note to file" for a Chatbook edit, which no path produces; the New-note view "takes the stage" at 235 columns where it takes 105 of 235; the list "takes the width the empty work area would waste" beside a surviving 54-column pane; the chooser's promised back control, which is not rendered) and six missed (`/` outside the navigator; the footer naming focus in Info; picker focus on the Folder-files door; "Saved appears once per view"; a reopened note's bare "Saved"; a callout's type running into its body). Plus ~30 changelog clauses inside user prose. B's own pass VERIFIED 17 further claims — Ctrl+N, Escape, footer strings, Import-once copy, Folder-files keys and Session Git end to end — so the page is mostly true; these are the exceptions. Task **32626**.

## What's Working

1. **The three-worlds copy, placed exactly where the decision is** — the empty list's database/Folder-files/Add-from-files sentence, Folder files' "Nothing is copied into the Library", and the per-file frontmatter promise. Three sentences doing most of the information architecture.
2. **The Import-once review.** Outcome-grouped with exact counts, a 45-file run collapsed to one openable row, a reason on every skip, "not imported: date, mood" for dropped frontmatter keys, and a receipt whose numbers hold up against the vault. Import once did not touch a byte of the source folder (`git status --porcelain` empty, `find -newer` empty).
3. **Delete → receipt → Undo.** Inline prompt in place, Cancel focused, footer naming the focused button, a named receipt with Undo focused, and the rail count restored on undo.
4. **Session Git's commit review** — What / Where / Impact / Recovery, with identity, branch, parent, "hooks will not run", "will be unsigned" and "no unrelated staged content will be committed", verified against `git log`. Better consent design than most desktop Git clients.
5. **The `state · Next: <what to do>` grammar**, used consistently across the surface, and shape-based focus indicators that survive a monochrome capture.
6. **Performance.** 0.66–0.70 s to open a 35 KB note, 0.44 s to review 65 sources, 0.56 s for a sync check, autosave committed to the DB in under 0.03 s. Nothing here is slow.

## Counts

- A: 1 P0, 5 P1, 4 P2, 2 P3, 45 captures. B: 97 checks — 73 PASS, 8 FAIL, 4 PARTIAL, 12 not-exercised — 20 defects (0 P0, 2 P1, 7 P2, 11 P3), 66 captures.
- Deduped: **24 tasks** — 5 high, 10 medium, 9 low — ids **32604–32627**. Nine findings mapped to existing open tasks, one to a recorded decision, two ruled out of scope.
