# Known context: Library + Library ▸ Notes (triage-stage pack)

- **Code under review:** worktree `.worktrees/notes-library-ux-review` at origin/dev `2d34cbf80d` (2026-10-02 18:59 -0700).
- **Audience:** the *triage* stage only. Fresh-eyes reviewers must **not** read this file before their walk. Its job is to tell triage which findings are already known, owned, fixed, or deliberate.
- **Sources:** the two latest critique snapshots (Library #10 `2026-09-11T06-13-11Z`, Notes #4 `2026-09-15T06-47-45Z`); `backlog/tasks` (4,779 files, 1,184 matched Library/Notes terms by title, label or body); `backlog/decisions` (253 ADRs); `Docs/User_Guide/library.md` + `library/*.md` (9,093 lines).
- **Method:** each prior issue was checked against HEAD code by grep plus a read of the cited lines, and cross-checked against the owning task's Implementation Notes and the commits that cite it (`git log --grep=task-N`). No live run was made for this pack, so **"FIXED" means fixed in code at HEAD, not live-verified.** Line numbers are at `2d34cbf80d`.
- **Status vocabulary:** FIXED (the code at HEAD carries the fix), STILL PRESENT (the defect is in the code at HEAD), PARTIAL (some of the issue is fixed, some is not), UNCLEAR (code alone cannot settle it, so it needs a live probe), BY DESIGN (a recorded decision keeps the behaviour), NOT A DEFECT (the owning task ruled it out with evidence).

---

## 1. Prior critique issues: status at HEAD, with the owning task

### 1a. Library critique #10 (2026-09-11, 25/40, dev `1f3184655b`)

The critique-10 wave filed tasks 32346–32366, all Done. Its riders 32378–32392 are mostly still **open**.

| # | Prior issue (sev) | Status at HEAD | Evidence (file:line @ 2d34cbf80d) | Task |
|---|---|---|---|---|
| L1 | Footer drops canvas key hints whenever an Input has focus (P1) | FIXED | `UI/Screens/library_screen.py:4748-4800`: the footer now reads `typing in field \| esc <label> \| after esc: <verbs> \| … \| F6 next pane`. The task found the premise half-wrong: the verbs are *not* live while typing (a focused Input consumes printable keys, ADR-031 r3), so they are listed after `after esc:` and not re-advertised as live | 32346 |
| L2 | Media row age `audio · 10m` reads as a duration (P1) | FIXED | `Library/library_media_state.py:961` `media_updated_age_copy`. The row says **"updated 10m"**, not "added …" (commit 2c22e5d3fe: the age is `last_modified`) | 32347 |
| L3 | Media viewer Find unreachable by keyboard; a failed attempt arms `t` trash (P2) | FIXED | `library_screen.py:1202` `Binding("ctrl+f","library_media_reader_find")`; `Library/library_media_viewer_state.py:579` gate: "This tab has no text to search · switch to Read or Analysis." | 32348 |
| L4 | New profile with a pre-written config lands on the full rail, with "Back to Get started" orphaned (P2) | FIXED in code; **docs stale** | `library_screen.py:21515-21541`: an unstored EXPANDED falls back to UNKNOWN, which resolves to STARTER when all evidence is EMPTY. `library.md:41-51` and `:927-937` still say such a profile "never sees" Get started, which contradicts the stamp at `:995-998` | 32349 (32059) |
| L5 | Media filter draft and applied filter disagree with no scope line (P2) | FIXED | `Library/library_media_state.py:887,1270-1296` (`scope_line` built from the APPLIED scope); rendered at `library_screen.py:19752` | 32350 |
| L6 | Import names a batch after a subdirectory; the landing under-reports a failed import (P2) | FIXED | `Library/library_ingest_state.py:2697-2707` (`os.path.commonpath` of member parents) | 32351 (follow-ups open: 32387) |
| L7 | "Collections" opens "Quick Capture"; its empty state blames unset filters; the pager is live at 0 of 0 (P2) | FIXED | `Widgets/Library/library_collections_capture_reader.py:397` `#library-collections-header`; two empty states (`collections.md:88-94`); pager at `:645` | 32352 (+32057 decision) |
| L8 | Export defaults to `quality: thumbnail` and never lists the bundle (P2) | PARTIAL | Default fixed: `Library/library_export_state.py:68` `DEFAULT_MEDIA_QUALITY = "original"`; preview list at `Library/library_export_scope.py:227`. **But the quality knob is inert end to end:** `Chatbooks/chatbook_creator.py:1563-1569` takes `quality`, and the only use is writing it into manifest metadata at `:1673`. Sizes print in KB only (`library_export_state.py:100`) | 32353 Done; **32381, 32382 open** |
| L9 | Single-page pager chrome survives on Skills, Collections and Trash (P2) | FIXED | `library_skills_canvas.py:1370`, `library_collections_capture_reader.py:645`, `library_media_trash_canvas.py:460` all route through `library_pager_layout` | 32354 |
| L10 | Rail collapses to an unlabelled `--->` grip while a note editor is open at 235 cols (P2) | FIXED (labels) / BY DESIGN (collapse) | `Widgets/Library/library_adaptive_reader_shell.py:34-37` `LIBRARY_PANE_GRIP_NAMES = {"Library": "Nav"}`. The collapse is the Notes work-session rule (`library_notes_controller.py:2472`, ≥120 editor cols, cancelled for the visit by one manual expand) and is documented at `library.md:313-318` | 32355 |
| L-m1 | `Loaded ·` leaks into a selected row's title | FIXED | `Widgets/Library/library_media_canvas.py:188` `_media_row_label_rest` (the state word moved to the fact line) | 32364 |
| L-m2 | Conversation rows use ` - ` where siblings use `·` | FIXED | `Library/library_conversations_state.py:220-226` | 32364 |
| L-m3 | "New Chat" reads like a button | STILL PRESENT | `library_conversations_state.py:197-203` maps only a null/blank title to "Untitled conversation"; the stored literal "New Chat" (written at creation by Chat services) passes through | **32379 open** |
| L-m4 | Prompt rows: "Prompt · Local ·" and schema-speak "System + User" | PARTIAL | `Library/library_prompts_state.py:2150-2154` now mixes "has system and user text" with "System only" / "User only". Console keeps a fourth vocabulary | **32378 open** |
| L-m5 | Stored analysis renders as raw Markdown | FIXED (render) / **security residue open** | `Widgets/Library/library_media_viewer.py:977,1129` (`looks_like_markdown_content`). The sink is unsanitized: `Widgets/Library/library_media_content.py:320-325` passes stored text straight to `Markdown(...)`. Find on a rendered Read tab marks nothing | 32365; **32392, 32384 open** |
| L-m6 | `○ Find` on Analysis and the Export disabled reason have no adjacent reason | FIXED | `library_export_state.py:255` `submit_blocked_reason`; viewer gate `library_media_viewer_state.py:579` | 32362 |
| L-m7 | Import footer `enter start` (the first Enter validates, the second runs) | FIXED | `UI/Library_Modules/library_ingest_controller.py:998` gives "start import" / "check this path" | 32364 |
| L-m8 | Trust-approved skill reads "needs review" with no precedence explanation | FIXED | `Library/library_skills_state.py:843` `skill_trust_header_line` (banner states the precedence) | 32363 |
| L-m9 | Conversations reader gets ~48 of 235 cols | FIXED | stale layout re-resolve in `LibraryConversationReader.sync_state` | 32361 |
| L-m10 | Search/RAG Sources panel triple-spaces four checkboxes | UNCLEAR, unowned | no task found; needs a live look | — |
| L-p1 | Jordan: "Draft — not saved yet" on a listed note | FIXED | `UI/Library_Modules/library_notes_controller.py:1442` gives "Empty note — type to keep it" | 32358 |
| L-p2 | Jordan: Details panel "Handoff · 0 eligible · 1 blocked", "Server sync WIP" | FIXED | `library_screen.py:14697` ("N item(s) can't be used in Console yet · …"), `:14807` | 32357 |
| L-p3 | Jordan: no in-product definition of RAG, Skill, Collection, workspace | STILL PRESENT (partial) | Rail glosses exist only for Media/Prompts/Skills/Search. The Collections gloss "saved captures" needs 34 cells, so it is hidden at the default 33-cell rail (`library.md:247-251`). "Workspace" is never defined in Library | unowned |
| L-p4 | Alex: `Use as source` disabled on all six conversations | FIXED | link-on-use, `Widgets/Library/library_conversation_reader.py:49` | 32107 |
| L-p5 | Alex: `]`/`[` type into the filter box | BY DESIGN (ADR-031 r3) | mitigated by L1's `after esc:` footer | 32346 |
| L-p6 | Alex: Conversations footer offers no canvas keys | FIXED (status still In Progress, AC ticked) | commits 10711f19b9, 436431ef38 | 32228 |
| L-p7 | Sam: active vs focused rail rows identical | FIXED | `css/screen_agentic_library.tcss:680-684` `.library-rail-row:focus { border-left: thick $ds-action-focus }` | 32359 |
| L-p8 | Riley: 60x24 clips copy mid-word with no ellipsis | FIXED | — | 32360 |
| L-d | Docs vs live: 14 contradictions | FIXED as a set; **new stale claims found, see §5** | — | 32366 |
| L-q | Open questions: why a landing page; who introduced "workspace"; is Collections a product; why the footer is the first thing sacrificed | UNCLEAR, design questions | landing capped at 96 cells (task-32217); Collections decided as capture+reading (ADR-113, task-32057) | — |

### 1b. Notes critique #4 (2026-09-15, 25/40, dev `77eb2601a6`)

The critique-4 wave filed tasks 32604–32627. All have merged fixes (PRs #2695, #2697–#2702), but **eight remain "In Progress"**: 32607, 32608, 32609, 32613, 32614, 32615, 32617, 32625. Six of them have every AC ticked. 32608 AC#3 (a captured keyboard-only end-to-end walk) and 32613 AC#2 (Tab-stop count stable across modes) are unticked.

| # | Prior issue (sev) | Status at HEAD | Evidence | Task |
|---|---|---|---|---|
| N1 | **P0** A `⇄ Both ways` root reads "✓ Up to date" while a Chatbook edit never reached the file; the manual Check had no Review door | PARTIAL | Editor-save path fixed: `UI/Library_Modules/note_session_port.py:165-190` calls `note_changed`, defined at `Notes/notes_sync_runtime.py:3232`. Check now opens the review: `library_notes_sync_controller.py:1500-1508` ("Manual check finished. N change(s) to review." / "Nothing to review."). **17 other note-write paths still send no signal** (new note, Console Save-as-Note, Research, ingest, MCP/agent tools, `library_save_note`, Import once over an existing note, chatbook import, delete/restore), and the row keeps reading "✓ Up to date". Documented honestly at `notes.md:802-822` | 32604 Done; **32633 open** |
| N2 | P1 Lasting sync plans a fresh create for every note Import once already made | FIXED (one direction) | `Notes/notes_sync_reconciler.py:74-78,439` skip with "Already imported by Import once — left as it is". **The mirror is open:** Import once on an already-synced folder still duplicates (`notes.md:1193-1196`). The imported notes are recognised, not adopted, so they never sync | 32605; **32637, 32636 open** |
| N3 | P1 "Choose File Notes Folder" opens with no keyboard focus | FIXED | `Third_Party/textual_fspicker/base_dialog.py:428` `RETURNS_A_FOLDER`; `:750` `_focus_initial_widget` lands on the path field | 32606 |
| N4 | P1 Info footer says "enter run action" for every control incl. Delete | FIXED in code; **docs stale** | `library_screen.py:8374-8394` substitutes the focused control's own label. `notes.md:516-518` still says Info's footer "is fixed … enter run action", which contradicts `:523-526` | 32607 (In Progress, 5/5 AC) |
| N5 | P1 "Activate reviewed root" and the Session Git commit buttons unreachable by keyboard | FIXED (pane-level); e2e unverified | `library_screen.py:9046-9048` `_LIBRARY_WORK_PANE_TAB_VIEWS` now includes the full-canvas views. AC#3 (captured end-to-end keyboard walk) unticked. Related: Manage sync folders is ~23 Tabs away | 32608; **32585 open** |
| N6 | P1 `/` focuses rail search outside the navigator while the footer advertises it | FIXED | `library_screen.py:8337-8350` (the tier drops printable chips while typing) | 32609 |
| N7 | P1 Every sync root row is titled "Sync folder (name unavailable before cutover)"; no path shown | **STILL PRESENT** | `UI/Library_Modules/library_notes_sync_controller.py:819` hard-coded literal. Documented as a known defect at `notes.md:770-777` | **32451 open** |
| N8 | Wave-4 regression: garbled Obsidian-toggle sentence | FIXED | `Widgets/Library/library_notes_add_from_files_canvas.py:473` | 32610 |
| N9 | Wave-4: false "More below — scroll." hint; stale "Resolution history unavailable" after activation | FIXED | `library_notes_add_from_files_canvas.py:310-334` (both display-managed from real state) | 32610 |
| N10 | Wave-4: skip granularity differs between Import once and the sync review; ~3 rows per file | FIXED | — | 32625 (In Progress, 4/4 AC) |
| N11 | H1: "Saved" printed twice four rows apart; two "Next:" lines in one pane | UNCLEAR, task open | Two producers remain: `Library/library_notes_session.py:449` ("Saved HH:MM", status line → authority line `Widgets/Library/library_notes_canvas.py:1268-1286`) and the status channel "Saved" at `library_notes_canvas.py:830`. `notes.md:464` claims "In Info, 'Saved' appears once" | **32514, 32513 open** |
| N12 | H2: "Use file_notes" as a button label (config key as label) | NOT A DEFECT | `Widgets/Library/library_file_notes_workspace.py:2964` builds `f"Use {_folder_label(...)}"` from the configured folder's basename (`:374-376`). The capture's fixture folder was literally named `file_notes`. Pinned by `test_use_folder_offers_the_modern_file_notes_root` | 32623 ruling |
| N13 | H2/H4: three folder pickers, one labelled "File name" while it picks only folders | FIXED | `UI/Library_Modules/library_notes_controller.py:5121-5144` pushes `SelectDirectory`. **Residue:** the picker footer advertises a dimmed `^s Select this folder` that no folder-only dialog can run (documented at `file-notes.md:144-149`) | 32611; **32647 open** |
| N14 | H4: four wordings for "go back" | FIXED (Notes surfaces) | `#notes-sync-back` uses `back_cue_label("Notes")` (`library_notes_add_from_files_canvas.py:1044,1118`). A bare "Back" survives in Session Git's push footer (`library_file_notes_git_panel.py:1800`), which may be intentional (`file-notes.md` Esc table: "push review → Back") | 32624, 32553 |
| N15 | H5: per-row `☐ Skip` / `☑ Create new` are mutually exclusive but drawn as two checkboxes | **STILL PRESENT** | `Widgets/Library/library_note_import_canvas.py:64-73` (`_choice_label` uses `LIBRARY_GLYPH_SELECTED/UNSELECTED`, i.e. ☑/☐) and `:1131-1146`. Radio glyphs `●/○` exist (`Library/library_shell_state.py:141-143`), and the guide's own legend says a one-of-N choice must not be drawn as checkboxes (`library.md:346-351`). Commit 7d2db0871d (task-32235) chose the checkbox pair deliberately | unowned (32235 made the choice) |
| N16 | H6: root row has neither name nor path | STILL PRESENT | same as N7 | 32451 |
| N17 | H8: Info leaves ~25 of 40 rows blank; the chooser stacks three headings | FIXED | Info became a two-column properties layout; the chooser asks once | 32642, 32612 |
| N18 | H9: a failure names its cause but not the next action | PARTIAL | The sync Check is fixed (N1). The generic `"Review the error, then keep editing."` survives at `library_notes_canvas.py:1269` for export, copy and import failures | **32573 open** |
| N19 | H10: ~30 "(Was … superseded by task-…)" changelog clauses in `notes.md` user prose | PARTIAL | **12** remain in `notes.md`, **8** in `file-notes.md` (grep `(Was \|superseded by task`). Example: `notes.md:134-137`, `:743-745` | 32626 Done |
| N20 | "Partly fixed" from the #4 like-for-like: Preview double title (frontmatter), word count 5,453 vs 5454, back wordings | FIXED | — | 32620, 32623, 32624 |
| N21 | Docs vs live: 4 new + 6 missed | FIXED as a set; **new stale claims found, see §5** | — | 32626 |

---

## 2. Open Library / Notes tasks (To Do · In Progress · Blocked)

Curated from 258 open matches down to 193 Library/Notes-relevant tasks (ruff debt, backup/recovery, CI, Console-only and unrelated media generation removed). There are **no Blocked tasks**. "IP" means In Progress. One line each, grouped for triage lookup.

### 2a. Owned defects most likely to be re-found by a UX walk

| Task | St | What it covers |
|---|---|---|
| 32451 | To Do | Manage sync folders: every root titled "Sync folder (name unavailable before cutover)", no path |
| 32633 | To Do | 17 note-write paths (new note, Console save-as-note, tools, imports, delete/restore) send lasting sync no signal; the row says "✓ Up to date" |
| 32637 | To Do | Import once on an already-synced folder creates a second note per file |
| 32636 | To Do | Lasting sync recognises Import-once notes but does not adopt (bind) them, so they never sync |
| 32635 | To Do | Notes has no bulk delete (select mode offers only Export selected) |
| 32585 | To Do | Manage sync folders buttons are ~23 Tabs away |
| 32649 | To Do | The Library notes \| Folder files strip is reachable only by Shift+Tab; the Notes rail row has no footer chip |
| 32648 | To Do | Tab from the Folder files search field leaks into the Library rail, not the Files tree |
| 32647 | To Do | The folder-only picker footer advertises a dimmed `^s` it cannot run |
| 32514 / 32513 | To Do | Note editor reports save state twice in two vocabularies / move the save state onto the chrome strip |
| 32573 | To Do | Generic "Next: Review the error" for export/copy/import failures |
| 32578 | To Do | Three lasting-sync refusal reason codes reach the row with no copy |
| 32586 | To Do | Lasting sync has no folder-creation door of its own |
| 32587 | To Do | Obsidian toggle for a synced root is process-lived (needs device-schema v3 to persist) |
| 32575 | To Do | Editor heading strip clips a duplicate note's tie-break at 100x30 |
| 32503 | To Do | A saved note's row keeps its stale age until the next refetch |
| 32391 | To Do | Ctrl+N paints a transitional Create frame for ~1 s (invites a double press) |
| 31974 | To Do | Notes select-mode toolbar overflows the Items pane at every width |
| 32590 | To Do | Notes browse "Export" is the only export action without an ellipsis |
| 32588 | To Do | Import once's files branch joins up to three bare basenames on one line at 48 cols |
| 32656 | To Do | The two reviews share a collapse threshold but not a run order (interleaved folders collapse differently) |
| 32577 | To Do | Import receipt reader takes only the latest session item and drops non-IMPORTED/UPDATED outcomes |
| 32571 | To Do | `_clear_finished_library_notes_operation` drops a completed navigator receipt |
| 32501 | To Do | Folder files: Session Git unreachable (off-screen work pane) at the "supported" 40x20 floor |
| 32581 | To Do | `#file-notes-git-bulk-toggle` is 31 cells and cannot fit below ~36 cols |
| 32511 | To Do | Session Git entry focus can steal a chosen in-panel control |
| 32516 | To Do | Folder files delete-receipt dismissal race (NoMatches) |
| 32457 | To Do | FileOpen "Select folder" silently returns the parent dir when a folder row is highlighted but not entered |
| 32580 | To Do | fspicker `_select_file` does not select the filename it click-fills for a folder-offering FileOpen |
| 32651 | To Do | EnhancedFileSave never got FileSave's filename-field focus |
| 32652 | To Do | Media Import's Browse picker inherits `offer_select_folder`, never live-walked |
| 32653 | To Do | Enhanced picker family has no footer, so its modals show the host screen's chips |
| 32660 | To Do | File picker: Ctrl+R opens an empty panel when there are no recents |
| 32661 | To Do | File picker: the listing shows one content row at 100x30 |
| 32308 | To Do | File dialogs: retire or surface the hidden Ctrl+L path bar |
| 32579 | To Do | Media Import picker inherits Notes' arrival focus and focus cue, unwalked |
| 32650 | To Do | Media: Escape never reaches the rail from the Media list (alternates filter ⇄ list), although the footer says `esc focus rail` |
| 32379 | To Do | Conversations show the stored literal "New Chat" |
| 32378 | To Do | Prompt lane summary reads four different ways across Library and Console |
| 32381 | To Do | Export media-quality knob is inert end to end |
| 32382 | To Do | Export sizes KB-only ("about 1048576 KB") |
| 32383 | To Do | Media Trash filtered to zero says "Trash is empty" |
| 32384 | To Do | Find on the Read tab marks nothing on a rendered Markdown item |
| 32392 | To Do | Read/Analysis render stored text through an unsanitized Markdown sink (security) |
| 32387 | To Do | Import: two declined review follow-ups (batch name from stored metadata; lifecycle provenance key) |
| 32304 | To Do | Below 64 cols, Collections/Conversations/Skills give the whole stage to an empty work pane |
| 32309 | To Do | Glyph-legend follow-through: selection glyph literals in 6 files; "1 items" strings |
| 32293 | To Do | Trash type chooser marks its cursor by colour alone |
| 32301 | To Do | List-entry focus re-arms from a fixed 2 s window, not each list-arrived seam |
| 32307 | To Do | Media restore should preserve pre-delete `last_modified` (today it jumps to the top of Newest) |
| 31568–31584 | To Do | Media wave-5 riders: Reader exit at ≤92 cols opens Items; select mode survives an Escape hop; F6 ring starts on inputs; 100x30 `esc focus rail` to an unfocusable target; review-set chip wording; bulk Analyze cancel / survive leaving Library; Import focus across recomposes; transcripts with `#` lines default to Rendered |
| 31969 / 31970 | To Do | Reader grip width chosen outside the resolver; whether a typed custom Items width is obeyed everywhere |
| 31973 | To Do | Media preview pane is dead in production (delete or re-enable) |
| 31647 | To Do | Conversations/Notes/Prompts select-mode rows swallow a fast second click |
| 28010 / 28014 / 28017 / 28018 / 28021 / 28239 | To Do | Reader content box capped at 18 rows; rail media counts stale after Trash restore; Import run-level jump to imported items; no-provider hint points at raw config; match count undercounts per line; Prev/Next don't cross page boundaries |
| 28019 / 28022 | To Do | First-run wizard media-first path; command palette near-duplicate Library entries |
| 15130 / 15140 / 15370 | To Do | Media Trash permanent delete + Empty Trash (ADR-055 B); select toolbar overflows <110 cols; hybrid coverage copy |
| 2151 / 2520 / 2765 / 3601 | To Do | RAG answer clamp mid-word; landing footer pin drift; Home "Import Library sources" lands without ingest context; plain-profile RAG requires embeddings runtime |
| 20977 | To Do | Losing terminal focus fires the ingest URL probe with no gesture (privacy) |
| 20979 / 19195 | To Do | Flashcards viewing/SRS surface undecided; owner call on Study sidebar tooltips that overpromise |
| 33003.15 | To Do | Library pane frames measure 1.28:1, below the WCAG 1.4.11 3:1 floor |
| 33005.9 | To Do | Home, Library and the first-run wizard should read shared connection evidence |
| 32811.4 | IP | Library surfaces that show the wrong thing (speaker rename swaps preview; note location row elided to another pane's width; chunking-template picker shows "None" while a template is submitted) |
| 32236 | IP | Blocked RAG Answer panel repeats one sentence and names an env var. AC#3 open (one source for provider-gate reason/remedy) |
| 32235 | IP | One state-glyph legend (3/3 AC ticked; status not flipped) |
| 32228 | IP | Conversations footer `esc focus rail` and `/` (1/1 AC ticked) |
| 32607 / 32608 / 32609 / 32613 / 32614 / 32615 / 32617 / 32625 | IP | Notes critique-#4 fixes, all merged. 32608 AC#3 and 32613 AC#2 are unticked |
| 32185 / 32202 | IP | Notes folder-tree / compact note-region behaviours newly reachable (3 + 2 ACs open) |

### 2b. Collections / server parity (open, mostly roadmap)
30016 atomic Server capture hard delete · 30017 Server tag/domain facets · 30018 (+.1/.2/.3) Server Collections templates, digests, import/export · 30019 decide legacy Collections migration or retirement.

### 2c. File Notes writable programme (TASK-399.x, roadmap; only 399.1 is IP)
399.1 isolate file-note projection storage (IP) · 399.2–399.5 read-only root preview, search/reconcile, File Notes workbench, Database parity · 399.6–399.11 APFS substrate, journaled save/autosave, rename/move/conflict UX, delete with restore, protected files, coalesced history.

### 2d. Performance (open)
281 (IP) targeted sync_state updates · 33280 PERF-21 (whole-screen recomposes, ingest O(N²), Folder Files poll) · 33286 PERF-27 Notes sync data paths · 24457 eight SQLite connections per visit · 22660 virtualize the Reader Markdown view · 22888, 21243 remaining whole-screen recomposes · 21132 note-folder CTE · 21247, 21249 notes-sync runtime gating/residue · 21240 abandoned setup re-arms the notes-sync start gate · 31508, 31509 review-set 2N+1 / Trash re-measure · 32804.4 / 32804.9 / 32804.11 (IP) Reader memoise, semantic-query offload, per-keystroke costs · 32803.3 (IP) media cleanup cutoffs.

### 2e. Test health / architecture (open; context only, not UX)
31249, 32199 (IP), 32462 (IP), 32386, 21232, 23145, 23149, 23152, 31422, 31881, 31971, 31975, 32105, 32171, 32298 (IP), 32310, 32455, 32456, 32568–32570, 32574, 32576, 32582–32584, 32654, 31424, 31650, 31651, 32013, 32088, 32089, 32097, 32170, 31421, 31423, 31572, 31573, 31574, 31577–31580, 31630, 31972, 32292, 32305, 32634.

### 2f. Undecided idea backlogs
32559 (critique #3 improvement ideas: one review renderer for both import paths, etc.; never ruled). 32627 ruled critique #4's ideas: 32640–32643 accepted and Done; "fold the three-worlds decision into one screen" DEFERRED.

---

## 3. Recently done (since 2026-09-11)

**Library critique-10 wave (2026-09-11), all Done:** 32346 footer `after esc:` · 32347 media row age "updated N" · 32348 Find ctrl+f + gate · 32349 empty profile → Get started · 32350 media scope line · 32351 import batch name + landing failure count · 32352 Collections header/empty states/pager · 32353 export default original + bundle preview · 32354 shared single-page pager rule · 32355 grip labels "Nav" · 32356 n/Ctrl+N create a blank note directly · 32357 Details plain-language copy · 32358 "Empty note — type to keep it" · 32359 rail focus bar · 32360 <64-col return + no mid-word clip · 32361 Conversations reader width · 32362 inline disabled reasons · 32363 skill trust precedence banner · 32364 copy polish · 32365 analysis renders Markdown · 32366 guide reconciled (14 claims) · 32388 workspace-hop Undo receipt · 32389 Notes below 64 cols · 32390 dead CSS · 32393 + 32461 Prompts dirty vetoes speak · 32302, 32303, 32306 conversation focus / RAG glyphs / Details wrap decision · 32462.1 workspace handoff refresh · 32464 Console library-access radio glyph.

**Notes critique-#2/#3 waves (closed 2026-09-11 → 09-14):** 32143/32146 (ideas: editor chrome strip; capture a Console answer as a note) · 32184, 32186, 32201, 32215, 32217, 32218, 32233 · 32242–32272 (critique-notes-2026-09: lasting-sync Check crash, egid admission, Session Git contract, preview cap, import review pagination, path concatenation, duplicate-title tie-break, Undo into a collapsed folder, primary actions 20–38 rows below, 12.6 s open, a11y polish, frontmatter/wikilink fidelity, guide sweeps) · 32294, 32295 · 32467 (late backlinks crash) · 32518, 32519 (activation leaves list stale; Resume lands Failed) · 32533–32558 (wave 4: InvalidSelectValueError exit, "Up to date" beside "Manual check failed", unnamed review rows, Use-in-Console fail-closed, Preview focus footer, word count, focus after delete, Import once by keyboard, re-import duplicates, UTC clock, false "Git · N", toolbar clip, Retarget/Disconnect reason, landing focus marks, 60x24 New, duplicate ids, disabled reasons, `/` stale text, Preview title, Folder files polish, back wordings, import copy, first-run hand-off, whitespace title discard, "Add from" clip, guide residual).

**Notes critique-#4 wave 5 (2026-09-15):** 32604 note→file signal (editor path) · 32605 sync skips Import-once files · 32606 folder picker focus · 32610 garbled sentence / scroll hint / history line · 32611 folder-only picker for lasting sync · 32612 chooser three-worlds copy · 32616 list row updates title on save · 32618 Obsidian embeds stated · 32619 CSV failure names good rows + next action · 32620 Preview H1 left-aligned, callouts · 32621 Folder files "No file open." + non-note legend · 32622 receipt "View N imported notes" · 32623 typographic consistency · 32624 back wording · 32626 guide residual · 32627 ideas ruled · 32640 "where this note lives" row · 32641 vault recognised before review · 32642 Info two-column properties · 32643 vault-aware folder picker.

**Design-system journey reviews (2026-09-15 → 09-18):** Prompts 32602, 32603, 32629, 32630, 32632, 32638, 32750 · Skills 32646, 32655, 32657 · Collections 32658, 32659, 32662 · Import 32663–32667, 32696–32698 · Conversations 32701, 32703–32706 · Search/RAG 32712–32720, 32751 · Notes 32752 (empty folder actions visible), 32765 (compact next-step status) · 32598, 32599, 32600 (rail toggles focus, bounded queries, compact import assertion).

**Other:** 2376, 2377, 2530, 4111, 15390 (RAG handoff/scope/run-gate/Open no-op/test) · 18930 Console prompt stash against Library prompts · 31815 modal-inventory repair · 32107 link-on-use · 32800.2 crash on Escape in the Prompts work pane · 32801.1 Media reading-list re-import · 32804.2 settings fields reload per keystroke · 32810.5 P3 consistency bundle (Library widgets) · 32869–32872 Library Artifacts (Reports, Chatbooks, browse cutover) · 32881 Notes clients vs server release contracts · 33096 structured document validation design · 32690/32691 file-to-note workflows · 33125 docs policy: verification goes in task notes, not User Guide stamps.

---

## 4. ADR constraints on Library / Notes design

| ADR | Title | Rule that constrains a Library/Notes recommendation |
|---|---|---|
| 086 | Share an adaptive reader shell within Library destinations | Media, Conversations, Notes, Prompts and Skills share one 3-region shell: rail and list independently collapsible, **work pane permanent**; two full-height grips (persist open/closed only; widths change via Settings, not drag); Library collapses before the list; stable thresholds plus hysteresis; preferred vs responsive vs effective geometry |
| 084 (media-reader-ia) | Make Library Media an adaptive reader with a permanent Reader | NetNewsWire shape; Reader is the width priority and never a collapse target; 5-column grip per pane |
| 104 | Settle Library Media returns at current-owner geometry | Return focus/scroll restoration is event-driven, two-gate, authority-fenced. Don't recommend timer-based focus restores |
| 076 | Library Lifecycle Progressive Disclosure | No Beginner/Expert mode and no second onboarding wizard; rail lifecycle `unknown/starter/expanded/graduated`; graduation is permanent (deleting content never moves backward); Explore remembered; deep links bypass compact presentation |
| 067 | Library Top-Level Pagination Contracts | Source-owned, exact-total, bounded pages of **≤20**; no generic Library data controller |
| 055 | One reversibility rule for Library destructive actions | Pattern A soft delete: receipt + Undo + durable Trash; B hard delete: state permanence; C named blank-note GC (silent); D draft discard: confirm, no receipt. Single and bulk variants share one seam. **A Notes bulk delete (32635) must reuse the single-delete seam and receipt** |
| 031 (+refinements 1340, 16211, 16350, 32458, 33622.10; amended by 210) | TUI Keybinding and Footer-Hint Conventions | Reserved: Ctrl+Q, Ctrl+P, F1, F6. **Never bind** Ctrl+C/V/X/S/D/Z/A/R/W for app actions. Screen actions are single letters; destructive single letters need confirmation. Footer may advertise only working keys (advertised ⊆ bound, advertised == working in context). Every Library-reachable modal: safe Escape / backdrop / Cancel. **Known tension:** the skill editor binds `ctrl+s` (`library_screen.py:1047`; `skills.md` documents it), and Notes and Folder files deliberately have no save key |
| 034 (shared-rail-disclosure-glyphs) | Disclosure glyph ownership | `▸/▾` glyphs owned by `destination_rail`; one definition each |
| 011 | Chatbook Workbench UI System | Shared Textual-native workbench: stable composition, explicit state snapshots, visible workflow controls, contextual help |
| 015 | Complete the shell destination IA | Every route has a home and every screen says what it is; retired Notes/Prompts/Skills/Search/Ingest/Media route into Library |
| 003 | Settings Library/RAG Defaults Boundary | Settings owns only global `rag.*` defaults; per-run Library choices stay in Library (task-19647 records drift) |
| 113 | Separate Collections capture authority from Media and legacy containers | Collections = saved URL captures + reading lifecycle, not generic membership; one active authority (Local or Server) |
| 118 | Chunking Lab local execution and recovery | Library owns Chunking Lab (lazy screen, returns to opener); no global destination |
| 164 | Keep Library Re-chunk work and feedback for the app session | One ephemeral run object per app; panels subscribe and unsubscribe |
| 147 | Reversible conversation archive and exact resume | Archive ≠ delete ≠ workspace archive |
| 079 (console-library-conversation-authority) | Per-conversation Console Library authority | Manual Search Library, auto pre-send retrieval and assistant Library tools are three mechanisms with separate controls |
| 030 | Direct Local Library Tool Boundary | Agent/MCP Library reads via one descriptor-backed service. Note writes by tools bypass the sync signal (see 32633) |
| 172 | Browse Artifacts within Library | Artifacts (All, Chatbooks, Reports) live in Library and reuse the adaptive reader; Ctrl+6 routes into Library. Implementation pending |
| 021 / 029 (file-notes-disk-authority) | File-backed Notes disk authority | **Disk files are the sole content authority** for Folder files; frontmatter and bytes preserved; SQLite is a rebuildable replica, never an editor authority |
| 027 | Portable Database Note Session Coordinator | One coordinator owns the draft, baseline, version, dirty/saving/conflict state, and serialized saves. Don't propose a second save path |
| 035 / 038 / 039 | File Notes Session Git index, guarded commit, guarded push | Process-memory session owner; trust per process; commit only Chatbook-owned staged entries; push only the one exact guarded commit to its existing upstream; no fetch/pull/merge/credential management |
| 059 | Notes Folder Import and Device-Local Sync Ownership | Folders = entities + memberships; **manual** vs **sync-managed** membership classes; one lasting-sync root owns derived membership. Explains why synced notes cannot keep the vault tree (notes.md:1205-1209) |
| 073 | Notes Sync Round-trip and Interoperability Constraints | ≤1 active binding per note per device; path ↔ note 1:1; **no automatic winner** (explicit `KEEP_FILE/KEEP_NOTE/KEEP_BOTH/SKIP`) |
| 105 | Portable Notes organization and Agent Lessons | Six organization domains sync as one capability, never partially; local and portable IDs separate |
| 194 | Portable note language and local structured editing | `content_language` is canonical note metadata through every path |
| 097 (boot-budget-ratchets) | Boot budgets are ratchets | Any recommendation that adds imports/CSS to Library's boot path must defer or shed cost; ratchets never rise |
| 099 | Two ADR-099 files: Persistent Terminal Session runtime / Schedule editor stays modal (Proposed) | **Neither governs Library.** The schedule ADR is only a precedent for modal-vs-inline editing |
| design docs | `backlog/docs/design-library-row-state-markers.md` (APPROVED, Option A), `design-library-review-sets.md` (DRAFT), `library-decomposition-recipe.md` | Media row markers for analysed/reviewed; review sets as first-class objects; how `library_screen.py` (~36 k lines) is being split into controllers |

---

## 5. Docs claims checklist (docs vs live)

How to use it: each row is a concrete, testable claim with its doc line. **⚠ STALE/CONFLICT** marks rows already shown wrong by code or by another doc line at HEAD. Test those first. Unmarked rows are unverified.

### 5a. `Docs/User_Guide/library.md`

| Line | Claim | Pre-check at HEAD |
|---|---|---|
| 24-26 | Ctrl+3 opens Library; Ctrl+P "Tab Navigation: Switch to Library"; Ctrl+6 opens All artifacts | — |
| 27-37 | Typing notes/prompts/skills/ingest/research/media/search/study in the palette routes to Library; "Tab Navigation: Library — Skills" lands on Skills | — |
| 41-51 | "If the config file already existed … Library opens the **full rail** even on a completely empty profile, and Get started never appears" | **⚠ STALE**: code `library_screen.py:21515-21541` (task-32349) resolves to STARTER. Contradicts this page's own stamp at :995-998. Lines 927-937 repeat the stale claim |
| 53-62 | Get started rail = Import…, New note, Explore all tools; canvas = Import a file, Find it, Use it in Console, unlocking in order; blocked step says "Find it needs something to search — Import a file first."; "Checking existing Library content…"; "Retry source check" | — |
| 79-88 | Explore remembered; Back to Get started while still empty; graduation is silent (no toast) | — |
| 99-152 | ≥120 cols landing: Continue / Needs attention / From your Library (Notes → Media → Conversations) / Quick actions (Import…, New note, Search); focused recents row has the left bar; footer "enter open notes"; source-failure callout with Retry and "· attempt N"; 5 s deadline copy | — |
| 154-166 | The landing stays beside the rail at 100x30; only <64 cols is single-stage | — |
| 179-185 | Rail auto width 29–39 cells; item lists default 50; custom 24–48 | — |
| 187-209 | <64 cols: one stage; "‹ Library" or Escape returns; footer `esc back to Library` replaces `/` and `F6` | — |
| 211-215 | An empty work pane gives its width to the list (Prompts, Skills, Collections, Conversations); landing hub capped | — |
| 216-217 | Header reads "Library \| Local" (or "Library \| Server: <label>") | — |
| 221-228 | Collapse / **Nav** handle spells N/a/v downward; list grip names its pane (Items, Prompts, Skills, Folder files) | matches `library_adaptive_reader_shell.py:34-37` |
| 231-237 | `/` outside a text field jumps to Search Library…; a second `/` or a click selects a stale query | — |
| 238-257 | Four rail sections; glosses Media "your files", Prompts "reuse", Skills "AI add-ons", Search/RAG "find all"; Collections gloss hidden at the default 33-cell rail; short labels "Chats", "Cards", "Captures" | — |
| 258-273 | Study rows open a staging canvas ("This page shows what carries over"); Continue in Study leaves; Escape returns; "▾ scroll for more" | — |
| 276-277 | Import… and New note reachable with **i** and **n** | — |
| 278-303 | Footer chips per surface: landing "i import content", "ctrl+n new note"; Search/RAG "u use Library context in Console", "o open evidence"; list "esc focus rail"; editor "esc back to notes" + "enter save note"; Export "esc back to Media/hub"; `typing in field \| esc leave field \| after esc: …`; <64 cols single chip | — |
| 305-318 | Notes adds the "Library notes \| Folder files" strip; the note editor collapses the rail at ≥120 editor cols; a manual expand ends that for the visit | — |
| 322-353 | Glyph legend: `█` cursor; `☐/☑` toggled selection; `✓/✗/–` outcomes; `○` disabled beside a reason; `●/○` radio only in Console Library access + first-run wizard; `▸` current rail row; `⇄` two-option toggle; **"painting a radio as a checkbox would promise a multi-select"** | **⚠ CONFLICT** with Notes Import once rows `☐ Skip` / `☑ Create new` (`library_note_import_canvas.py:64-73`), see N15 |
| 377-411 | Empty-page copy for Media/Conversations/Prompts (source-empty vs filter-empty, with action buttons); single-page pager rule also on Skills/Collections/Trash | — |
| 415-427 | Media Trash: 20/page newest-first; Restore; Delete forever `x` (inline confirm, "cannot be undone"); `r` restores | Trash filtered to zero still shows the "empty" copy (32383) |
| 449 | Create ▸ New note opens the creation canvas (Blank note / From a template) | consistent with notes.md:589-592 (n/Ctrl+N skip it) |
| 464, 658-660 | Export disabled in server mode: "Export packages local content only." | — |
| 474-493 | Details collapsed by default; label click toggles; Diagnostics disclosure with DB sizes; Handoff "N item can't be used in Console yet · … · …"; continuation lines indented 2; "Everything here is stored on this machine · syncing to a server isn't available yet."; Create local workspace dialog; Use in Console tooltips | — |
| 500, 545 | "The three **Create** rows …" / "Select Study decks … in the **Create** section" | **⚠ STALE/CONFLICT** with :240 and :258 (Study is its own rail section) |
| 518-524 | Study screen header "Library ▸ Study", "Esc: back to Library"; Escape returns to the Study decks staging canvas | — |
| 528-532 | Rail search → Search/RAG grouped "Evidence · top 15 per source" | — |
| 537-540 | Create a note: "notes autosave (the meta line ends in 'saved')"; "**‹ Back to list** returns you" | **⚠ likely STALE**: the wide editor back cue is "‹ Notes" (notes.md:458), and the status line shows "Saved"/"Saved HH:MM" |
| 554 | `/` focuses the active list's own filter (Media, Conversations, Prompts, Notes); falls back to rail search on Skills/Collections/Search-RAG/Study | — |
| 555 | `u` only while the Search/RAG row is selected | — |
| 556-558 | ↑/↓ within lists (no wrap); Enter opens; Tab stays inside Library | — |
| 572-583 | Escape in any search/filter box hands focus to the canvas (nothing cleared); Escape on a plain list → rail's Search Library… (or Import… in Get started) | **⚠ CONFLICT** for Media: task-32650 (open) says Escape alternates filter ⇄ list and never reaches the rail |
| 584-593 | Media bulk-delete confirm: "Delete N selected items? You can undo right away, or restore later from Trash."; footer "esc cancel delete" | — |
| 594-618 | Editor Escape semantics: Notes autosave returns at once; Prompts dirty → "esc save or discard first" / "esc busy, try again"; media viewer forms take two Escapes, footer "back a step" | — |
| 620-621 | Escape and **Ctrl+S** bound inside the skill editor | **⚠ ADR-031 r2 conflict** (`library_screen.py:1047`) |
| 628-1072 | ~380 lines of "*Verified against …*" stamps (48 on this page) | **Docs hygiene**: dev CLAUDE.md now forbids User Guide stamps (task-33125) |

### 5b. `library/notes.md`

| Line | Claim | Pre-check at HEAD |
|---|---|---|
| 20-28 | A brand-new profile has no Browse section; three ways in: Ctrl+N / bare n (blank note + editor), rail New note row (New note view), Explore all tools | — |
| 50-61 | The first editable note in a wide session closes Library nav once; reopening keeps it open for the session; listed reset conditions; back cue "‹ " + destination everywhere | — |
| 63-67, 89-93 | Wide: strip keeps both switches; compact: strip becomes "‹ Library / Notes" | — |
| 94-116 | List rows "title · age"; duplicate titles add folder, then time of day, then `#xxxx`; the editor heading carries the tie-break wide; at 100x30 the heading drops it | 32575 open (clip at 100x30) |
| 117-127 | An empty work pane keeps a ~48-54 col floor ("Select a note to edit it here."); <64 cols the list is the whole stage | — |
| 127-137 | A rename that saves updates the list row immediately | — |
| 139-152 | One "Next:" at a time; tree newest-first; Sort Newest/Oldest/Title remembered; Sort disabled under a filter | — |
| 346-366 | Folder tree pages of 20; More folders / More notes / Load earlier / Retry per branch; "Locating note…" reveal | — |
| 381 | Filter status "filter: <text> · N results" | — |
| 382 | **New** opens the New note view; Ctrl+N skips it | — |
| 383-385 | Folder / placement actions and their disabled reasons ("This folder is managed by sync; …", "Unfiled is shown automatically; …") | — |
| 387 | "Sort: Newest" opens a choice strip; filtered shows "○ Sort: Newest" + "Sort unavailable — clear the filter" | — |
| 391 | Toolbar "**Export…**" opens the bundle canvas | **⚠ STALE/CONFLICT**: code label is bare "Export" (`library_notes_canvas.py:2040`), and import-and-export.md task 5 says so; 32590 open |
| 392 | Select/Done: "N selected", "Select all N shown", "Clear", "Export selected"; compact "Done/All N/Clear/Export"; "0 selected — Export selected unavailable"; the open note becomes a read-only preview with header "Read-only preview · Included / Not included in bulk selection" | 31974 open (toolbar overflow at every width) |
| 394-415 | `/` focuses and selects the filter; Tab ×1–4 from the filter: unfiltered New, Sort, Select, Add from files…; filtered New, Select, Add from files…, Export | — |
| 417-423 | Empty list "No notes yet. Create your first note."; "Agent_Lessons — where Console agents file reusable lessons (empty)" | — |
| 425-432 | <64 cols: status drops the "Library notes ·" prefix; no half-words | — |
| 458 | Back cue "‹ Notes" wide / "‹ Back to list" compact in Edit/Preview/Info | — |
| 459 | "Where this note lives" row ≥80 cols: "In the Library database only — no file on disk" / "In a synced folder" + path + file write time | 32811.4 (IP): the row is elided to the wrong pane's width |
| 460 | Keywords on their own row under the title, one Tab from Title | — |
| 461 | Preview takes focus (pgup/pgdn work at once); a duplicate `# Title` H1 is shown once; H1 left-aligned; callouts render as "Note: Title"; Escape leaves Preview for the **list** ("esc back to notes" / "esc notes") | — |
| 462-463 | Info: labelled Created/Modified/Version/Words rows in two columns wide; "Linked from (N)", "Linked from — checking…", "(0) — no notes link here yet", cap 50+ | — |
| 464 | Status: "Saved", "Saving…", "Unsaved changes", "Conflict — …", "Save failed — …", "Unavailable — …"; "Saved 12:47" local time just after a save; **"In Info, 'Saved' appears once"** | 32514 open (two producers) |
| 465 | Chrome strip "N words · L:C" in Edit at ≥80 cols, thousands separator, no Tab stop | — |
| 466-470 | Save; Use in Console (prompt "Use this note as context and help me work with it."); Copy toast "Note copied to clipboard as markdown!"; Export toast "Note exported successfully to <name>"; inline Delete "Delete this note? Undo will be available in the Notes list." with Tab confined to Cancel/Delete and footer "enter cancel"/"enter delete" | — |
| 472-477 | "Loading note…"; after ~3 s "Unable to load note — timed out after 3 s. Press Retry." | — |
| 479-483 | Autosave ~2 s after typing; meta line "saving…"/"saved"; conflict banner "This note changed elsewhere — Overwrite saves your text; Reload discards it." | lower-case "saving…/saved" conflicts with :464's "Saving…/Saved" |
| 493-496 | Notes has no Ctrl+S and no save shortcut | — |
| 502-505 | Editor Tab cycle: ‹ Notes, Edit, Preview, Info, Save, Use in Console, Title, Keywords, Body; F6/Shift+F6 leave; Ctrl+End/Ctrl+Home; Escape returns (from Info: back to the editor first?) | :505 says "one press, from Edit, Preview, or Info. From Info it goes back to the editor first", which contradicts itself |
| 512-518 | "Info's footer is fixed instead — 'enter run action' — except while its inline delete confirmation is open" | **⚠ STALE**: code `library_screen.py:8374-8394` (task-32607) names each control; contradicts this page's :523-526 |
| 528-535 | Preview: Tab out of the body lands on Edit, not "‹ Notes" | — |
| 549-552 | Blocked save: "Can't leave yet — fix the title or press Discard new note." | — |
| 554-567 | Delete receipt "✓ deleted · <title>" above Undo/Dismiss; Undo reopens a collapsed folder; receipt dismissed by Add from files / Folder files | — |
| 589-635 | Ctrl+N / n skip the New note view; Blank note → "Empty note — type to keep it"; untouched blank discarded silently; whitespace-only title discarded with "Empty note discarded"; 8 templates listed; at 235 cols view ~105, list ~62, rail ~37; focus parks on Blank note; footer "enter create note" | 32391 open (Ctrl+N flashes a Create frame) |
| 639-650 | Add from files: heading "Add files to Library notes."; Import once vs Keep a folder synced consequences; a third line points at Folder files; bar holds only "‹ Notes" | — |
| 652-670 | Sync review rows "path · effect · destination", headings with counts, "Syncs .md, .markdown and .txt only; …", collapsed "▶ Archive · 45 files · …", Skipped reasons | — |
| 672-678 | Each picker remembers its own last folder; fallback `[notes] sync_directory`, then home | — |
| 680-697 | Check changes refusal copies ("That folder is inside Chatbook's own data directory. …", "Another Chatbook window is using that folder", "That folder is already connected", line-endings, >10 MB) | 32578 open (three codes have no copy) |
| 699-702 | Stale review → "Check again"; server: "Unavailable — server sync-folder capability not installed" | — |
| 704-728 | Conflict: View comparison; Keep file / Keep note / Keep both / Skip for now; Apply reviewed; receipt Undo ≤30 days; Resolution history | — |
| 736-765 | Manage sync folders: "Manual check finished. N change(s) to review." / "Nothing to review."; "◌ Changes available · Next: Review changes"; Pause/Resume; "⚠ Needs attention · Check failed — <reason> · Next: <action>"; "○ Retarget unavailable — not in this release" | matches `library_notes_sync_controller.py:1500-1508` |
| 770-777 | Every root titled "Sync folder (name unavailable before cutover)" (known defect) | still true (`:819`, 32451) |
| 779-800 | Receipts: newest 20, "when · what happened · file · note", "Wrote note to file" after an editor save; "⚠ Sync stopped · Next: Check changes" if the runtime stopped | — |
| 802-822 | Only the editor save signals sync; new note, Console Save-as-Note, tools, Import once over an existing note, chatbook import, delete/restore do not | still true (32633) |
| 824-827 | Root controls name their Enter in the footer ("enter check changes", "enter pause") | — |
| 1090-1094 | Template flow: rail New note → From a template → Meeting notes | — |
| 1096-1107 | Import once flow: Add another file, Change selection, Clear, Check selection, Import selected items, Last import | — |
| 1109-1123 | Keyboard-only Import once: Tab to Add from files…, Enter; Import once focused; picker on path field; Tab Tab Select folder; **Ctrl+S** "does the same"; Tab×3 Check selection; review focuses Import selected items | Ctrl+S in the picker is a modal-scoped binding; folder-only dialogs dim it (32647) |
| 1131-1143 | Set up lasting sync, step 3 | **⚠ DEFECT**: **two step-3s**. The superseded "File name" instruction (:1131-1135) is still present above its replacement "Folder path" (:1136-1143) |
| 1150-1156 | Activate reviewed root; receipt "N applied · listed under Receipts"; Back shows the "⇄ Sync managed" folder without restart | — |
| 1173-1196 | Import-once then sync: "Already imported by Import once — left as it is", `0 safe`; files not synced; no bulk delete; the reverse order duplicates | 32636, 32635, 32637 open |
| 1236-1256 | Use in Console status lines ("Use in Console complete — Linked to <ws> · staged in Console." etc.); failure copy names the next step | — |
| 1258-1270 | Console More… → Capture as note; "Saved to Notes" receipt with Open note; keywords console, conversation:<id>, message:<id>; blocked in temporary chats | — |
| 1283-1286 | After delete, focus on Undo, footer "enter undo delete" | — |
| 1288-1303 | Recently deleted (N) under the tree; Restore or `r`; Escape back; holds 20 | — |
| 1309-1327 | Keys: n / Ctrl+N, `/`, `g` folder tree, `e` export selected (select mode, only when something is checked), Escape focus rail, `r` restore; footer `n new note \| / find note \| g go to folder \| esc focus rail`; select mode `enter select note \| e export selected \| esc done` | — |
| 1388-1389 | "Notes rows have no ▸ marker" | — |
| 1396-2353 | ~950 lines of "Verified against" stamps (73) and 12 "(Was … superseded …)" clauses | Docs hygiene (see N19) |

### 5c. `library/file-notes.md`

| Line | Claim | Pre-check |
|---|---|---|
| 22-37 | Folder files via the source strip; "Opening File Notes…"; Escape climbs: editor → Files (`esc files`) → Library notes (`esc notes`); a switch saves unsaved edits first | — |
| 39-50 | First editable file in a wide session closes Library nav once; compact "‹ Files" | — |
| 57-73 | Before a folder is linked the rail has no grip; <~120 cols the rail is collapsed | — |
| 70-82 | Folder link row: "Choose a notes folder." + Details + Choose folder…; "Folder files edits Markdown files in a folder on disk, in place. Nothing is copied into the Library."; "Use <folder>"; "Linked · Local folder: <folder>" / "Checking …" / "Offline …"; Change… | — |
| 83-89 | Authority line "Folder files · Folder: <folder>" + "· Git · N change(s)" only for a confirmed git repo, else "· N session change(s)" | — |
| 90-110 | Navigator: New, "File contents…", Files tree, "Lists .md, .markdown, .txt and .text. Other files stay on disk."; dot-folders hidden; Load more per 100 | — |
| 111-117 | Work area: "No file selected", Idle/Dirty/Saving/Saved/Conflict/Error, Edit/Manage; "No file open." with no body box | — |
| 124-136 | Keyboard linking: Shift+Tab to the strip, Enter; Tab to Choose folder…; picker opens on a focused, selected "Folder path"; "Tab then Enter on **Select**" | the button is "Select folder" elsewhere (notes.md:1116, :1137); strip reachable only by Shift+Tab (32649) |
| 138-149 | The picker shows its own footer starting `esc Cancel`; at 100 cols it ends near `f5 Refresh direc`; `^s Select this folder` dimmed | 32647 (dimmed advertised key) |
| 421-452 | Link, create (`ideas/today.md` via New file path → Create), search, resolve conflict (Compare / Resolve conflict), Session Git: Review session changes (N) → trust → Stage all (N) → Commit staged (N) → Subject → Review commit → Confirm commit → Review push (1 commit)… → Endpoint Details → Authorize and check → Push 1 commit; "Check remote again — no push"; Restore a deleted file | 32501 (unreachable at 40x20), 32581 |
| 453-470 | Keys: Ctrl+End/Home (`ctrl+end end of file`), Esc ladder table, Session Git Up/Down/Tab/Enter, Esc never blocked by a running folder change; no Ctrl+S | — |
| 497-504 | Frontmatter hidden: "N lines of YAML frontmatter above this body are hidden here and kept exactly as they are on disk" | — |
| 505-509 | Caps 8 MB / 2,000,000 chars; >200,000 chars opens read-only with a 100,000-char excerpt; Export exact copy | — |
| 514-516 | Trust dialog "Trust repository for session changes?" returns every restart | — |
| 543-545 | No "Use in Console" for Folder files | — |

### 5d. Other child pages (keys and flows)

| Page:line | Claim | Pre-check |
|---|---|---|
| media-and-conversations.md (Keyboard) | Escape ladder chip readings `esc close` / `esc focus Items` / `esc focus Library` / `esc back`; in three-pane layout the rail row drops the esc chip; F6 cycles Library → Items → Reader content (heavy border); from an Items row: `]`/`[` items, `l` read-later, `c` Console, `t` trash, `s` select; review set: `]` "finish review", `m`, `R` | 32650 says list Escape never reaches the rail |
| media-and-conversations.md:388-393 | Empty states "No media in your Library yet. Import something to see it here." / "No media of type 'pdf'." / "No media of type 'pdf' matched “day2” in titles, content or keywords." | — |
| media-and-conversations.md:308-311 | Restore marks the item changed now, so it jumps to the top of Newest | documents behaviour 32307 wants changed |
| media-and-conversations.md:315 | Trash empty: "Trash is empty. Items you delete from Media land here." and "○ Restore" with a reason tooltip | filtered-to-zero shows the same copy (32383) |
| media-and-conversations.md (Common tasks) | Find: "Match 1 of N", "◀ Prev"/"Next ▶" under the box; Highlights "Add highlight" with ● swatch; Conversations Use as source links-on-use + Undo link; Select → Export selected | 32384 (Find on rendered Read) |
| prompts.md:81,250,278-285,472 | Sort strip Newest/Name; "New prompt · Unsaved changes"; dirty editor blocks Back/Escape/row/rail with footer **esc save or discard first**; "Saved." after Save Prompt | — |
| prompts.md (Common tasks) | Import a folder via "File or folder path…" (Browse… picks single files only); Use in Console appends User text; System lane only with explicit authorisation; Duplicate "(copy)"; Export… `Prompts · N items`; collection chooser + Manage collections + Apply memberships | — |
| prompts.md (Keyboard) | Enter in filter applies the debounced search; Escape cancels the collection manager | Prompts has no save key and no autosave (cf. Notes autosave, Skills Ctrl+S) |
| skills.md:62-63 | Empty "No skills yet — use Create ▸ New skill in the rail, or Import skill… above." | — |
| skills.md (Keyboard) | **Ctrl+S** saves only when new or changed; Esc closes More then returns to list | ADR-031 r2 conflict |
| skills.md:443-447 | "The Chunking Lab strip … paints under the header on every Library canvas … Escape does not leave it … tracked as task-32064" | **⚠ STALE**: task-32064 Done; the Lab lives in Details ▸ Actions with an Escape route (library.md:372; file-notes.md was corrected, skills.md was not) |
| skills.md:524 | Trust banner: with no trust store every skill reads "needs review" | matches 32363 |
| collections.md:88-94 | Filtered empty "No captures match these filters · clear them …"; unfiltered "No saved captures yet · press Quick Capture above …" | — |
| collections.md (Keyboard) | No screen keys; `/` → rail search; Escape → rail; F6 cycles | — |
| collections.md (Common tasks) | Quick Capture → Save capture; Mark Read; Move to Archive with Undo; Highlights; Link Note by exact Note ID; Filters; pages of 20; legacy "Export complete JSON…" | — |
| search-and-rag.md (Keyboard) | Enter runs; Tab visits Run, enabled source toggles, then cards (solid left block); PgUp/PgDn with a card focused; Enter on a card = select evidence; `o` opens; `u` stages (selects the focused card first); Esc in the query box hands focus to the panel; footer enter chip "run search" / "toggle Notes" / "switch mode" / "select evidence" | — |
| search-and-rag.md (Common task 6) | Click "**mode: Search ▸**" so it reads "mode: RAG Answer ▸" | **⚠ STALE**: code renders "mode: ✓ Search ⇄ RAG Answer" (`Library/library_shell_state.py:356`, `library_search_rag_panel.py:979`; library.md:339) |
| search-and-rag.md (Common tasks) | Sources toggles ☐/☑; Select evidence → Use in Console stages "Review evidence in Console"; Recent searches replay with current mode; Clear history; "Nothing in your library supports an answer to that." | — |
| import-and-export.md (Keyboard) | `i` opens Import from anywhere outside a text field; caret parks in the path field; footer `enter check this path` → `enter start import`; ⚠ warnings need Enter,Enter; `r` "Retry this batch"; Escape backs out of "Press Start again", else to the landing; F1 lists the same set | matches `library_ingest_controller.py:998` |
| import-and-export.md (Common tasks) | "Open in Library" after "✓ done"; folder scans stop at 1,000 files (" · more files not shown"); a pasted URL is not fetched before Start; "Copy install command"; Notes "Export" bare (no ellipsis); Prompts "Export…" `Prompts · N items`; Retry " · attempt 2" | Export size in KB only (32382); quality inert (32381) |

### 5e. Docs-wide hygiene facts (for triage, not a tester)
- "Verified against" stamps per page: library.md 48, notes.md 73, media-and-conversations.md 68, import-and-export.md 49, file-notes.md 26, search-and-rag.md 18, prompts.md 11, skills.md 11, collections.md 5 (**309 total**). Dev CLAUDE.md (task-33125) now says not to add stamps; existing ones were not removed.
- "(Was … superseded by task-…)" changelog clauses: notes.md 12, file-notes.md 8 (critique #4 called these out; task-32626 reduced but did not remove them).
- notes.md is 2,353 lines, ~40% of it stamps.

---

## 6. Cross-cutting tensions surfaced while building this pack (not in either critique)

1. **Three save models in one destination.** Notes autosaves with a visible Save button and no key. Prompts has no autosave and no save key, and dirty-veto blocks leaving. Skills has no autosave and binds **Ctrl+S**, which ADR-031 rule 2 forbids (`library_screen.py:1047`). Folder files autosaves.
2. **Radio vs checkbox legend broken by Notes import rows** (N15). The legend at `library.md:346-351` is the authority; the code chose the checkbox pair (commit 7d2db0871d).
3. **Stale docs that contradict their own later stamps**: library.md:41-51 vs :995-998; notes.md:516-518 vs :523-526; notes.md:1131-1135 vs :1136-1143.
4. **Status hygiene**: 8 critique-4 tasks are In Progress with merged fixes (6 with every AC ticked); 32228/32235 are the same. Triage should not treat them as open defects without a live re-check.
5. **Contrast**: Library pane frames measure 1.28:1 (task-33003.15), below the 3:1 component-boundary floor. This applies to every Library screenshot reviewers will judge.
