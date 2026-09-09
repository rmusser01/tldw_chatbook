---
target: "Library ▸ Notes sub-screen: create / edit / Obsidian import, first-timer and power user"
total_score: 16
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 8
timestamp: 2026-09-09T04-24-50Z
slug: w-chatbook-widgets-library-library-notes-canvas-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated scratch profiles and tmux sockets; parent re-tested every disagreement on a third profile and traced each surviving cause to code). Target `tldw_chatbook/Widgets/Library/library_notes_canvas.py` = Library ▸ Notes at origin/dev c4a7b1911f. Live at 235x52, 100x30 and 60x24; one empty profile per assessor (first-time journey) and one seeded profile (10 notes, 11 media, 6 conversations); a 71-file Obsidian-style vault fixture per profile; no LLM provider. Dev moved during the run: PR #2531 merged after the tip and addresses three findings below (marked).

Deterministic scan: `detect.mjs --json` returned `[]` (exit 0) on the canvas file and on `Widgets/Library/`. Unscannable, not clean: the detector reads web markup, not Python or Textual CSS. Browser visualization is not applicable to a PTY-drawn TUI; no overlay exists. Evidence = 118 tmux captures under the session scratchpad `notes-crit/{A,B}/caps/` plus the parent's re-tests.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of System Status | 2 | A folder change dies after 30 s with no message; a new profile has no empty state; a blank note reads "Saved" before a character is typed |
| 2 | Match System / Real World | 2 | A folder picker labelled "File name"; `.obsidian/*.json` reported as "Failed … could not be imported safely"; a note titled `{{date:YYYY-MM-DD}}`; "All planned items settled." |
| 3 | User Control and Freedom | 1 | Undo after delete is composed off-pane; a wedged Folder files session cannot be recovered without restarting; a wrong import source cannot be changed |
| 4 | Consistency and Standards | 2 | Two folder pickers with two field semantics and two start locations; `n` vs `ctrl+n`; `‹ Notes` vs `‹ Note`; Sort changes its label and nothing else |
| 5 | Error Prevention | 1 | Both pickers commit the directory being browsed, not the path typed; both assessors selected the source repository or the home directory by accident |
| 6 | Recognition Rather Than Recall | 2 | Two identical "Reading list" rows, no ages, titles clipped at 38 columns; strong: templates show the title they produce, review rows name their state |
| 7 | Flexibility and Efficiency | 2 | Fast opens and a real filter, but `/` types itself, Sort is inert, the 71-row review has no bulk action |
| 8 | Aesthetic and Minimalist Design | 1 | 145 empty columns beside a 38-column list at 235 wide; 71 five-line review rows; floating `--->` grips |
| 9 | Error Recovery | 1 | The timeout copy that exists in code never paints; the receipt says "11 skipped" with no list |
| 10 | Help and Documentation | 2 | The in-place three-worlds sentence is excellent; the guide is contradicted in 14 places, two behind "Verified against" stamps |
| **Total** | | **16 / 40** | **Poor** |

## Design Specificity Verdict

LLM assessment: the model is authored for this product; the surface is not. Authored: the canvas sentence that explains Library notes vs Folder files vs synced folders before anyone asks; the relationship-before-source gate; Import once that is provably byte-safe (sha of all 71 vault files identical after Check, after Import and after editing an imported note in-app); status strings naming source, authority, state and next step; repeat detection ("Unchanged repeat (19)") and folder-collision refusal on a second import. Category-interchangeable: a 38-column list beside 145 empty columns; a generic file-open dialog wearing a "Select folder" button; an importer that walks `.obsidian/` and `.trash/` and treats wikilinks, tags and frontmatter as plain text. The good thinking is invisible because the pixels do not carry it.

Deterministic scan: unscannable (above). Visual overlays: none possible.

## Overall Impression

The data model, the copy and the safety guarantees are better than the screen. Two silent-failure paths (the folder pickers and the poisoned Folder files session) and one geometry decision (the starved list pane) account for most of the lost points, and all three are narrow fixes. The single biggest opportunity is the Obsidian journey: today it copies a vault faithfully and adopts none of it.

## What's Working

- Import once keeps the local-first promise verifiably: 71 files hashed before and after; byte-identical through check, import and an in-app edit.
- The three-worlds sentence painted on the canvas pre-empts the exact confusion the architecture creates.
- Performance and integrity: the 35 KB note opens in 0.35 to 0.74 s; Escape while dirty saves; resize mid-edit loses nothing; focus is distinguishable by shape on every control B checked.
- The import review's grammar (per-item state, non-destructive defaults, unchanged-repeat detection, collision refusal) is the strongest thing on the screen.

## Workflow × persona

| Cell | What happened | Worst issue |
|---|---|---|
| First-timer · create | 9 interactions from wizard exit to first keystroke; the wizard and Get started point at provider setup; Notes (0) shows only an unexplained `Agent_Lessons` folder; a Blank note reads "Saved" before typing; the list row stays "Untitled" until you leave the field | No empty state, dishonest first save (P1) |
| First-timer · edit | Reopen is instant; Preview drops the title; Delete jumps the pane from Info to Edit and paints the prompt 14 rows lower; footer reads "enter confirm delete" while Cancel is focused; the receipt shows no Undo | Undo unreachable (P1) |
| First-timer · Obsidian | Chooser shows one button; Import once is 36 rows down; picker field says "File name", Select folder returns the browsed directory (the source repo); 71 five-line review rows; `.obsidian` "Failed (5)", `.trash` imported, `{{date}}` note, tags dropped, wikilinks literal; vault untouched. Folder files: picker opens at home, Select without Enter scans the home directory for 30 s, reverts silently, and every later pick wedges | Poisoned Folder files (P0) |
| Power · create | Ctrl+N works inside Notes (Blank note focused, 8 named templates); inert on the landing where the key is `n`; `/` inserts itself; toolbar wraps to three rows and hides Move/Remove | Accelerator inconsistency (P2, pinned) |
| Power · edit | 35 KB note fast; two identical "Reading list" rows with no age; Sort inert and "Title" never renders; burst title+Tab+body interleaves (partly fixed on dev) | Sort is a dead control (P1) |
| Power · Obsidian | Vault linked in under 2 s when it is the first pick; Keep a folder synced, Session Git, search and 45-note paging were not reached by either assessor (budget spent on the wedge) | Coverage gap, not a verdict |

## Priority Issues

- **[P0] A folder pick that does not finish poisons Folder files for the rest of the session, and the picker makes that pick likely.** Why: reproduced by me three times on a clean sequence: vault → `file_notes` links instantly; the home directory times out at 30 s; the next small folder then wedges too. Both assessors hit it because the picker opens at the home directory and its Select button returns the directory being viewed, so a typed path that was not submitted with Enter selects the home folder. The timeout copy defined in code ("Folder change timed out · previous folder kept…") never painted at 0.5 s sampling. Cause: PROVEN. `_change_root_with_deadline` cancels the asyncio task only; `set_root` runs `service.scan` in `asyncio.to_thread` under `operation_lock=self._service_lock`, so the scan thread keeps running and the next `set_root` waits behind it. B's "no folder can be linked" is retired: a clean first pick links in under 2 s. Fix: make the scan cancellable (check a cancel flag per directory or run it in a subprocess), bound the first scan by entry count with a "Keep waiting / Choose another" prompt, and paint the timeout copy. Suggested: /impeccable harden.
- **[P1] Both folder pickers commit the directory being browsed, not the path you typed.** Why: A imported the entire source repository; B did the same; both saw only a basename ("1 folder selected: notes-review"). Cause: PROVEN. `EnhancedSelectDirectory._select_viewed_directory` returns `_dir_nav().location`; the Import picker is `FileOpen(offer_select_folder=True)`, whose "File name" field is never read by "Select folder". Fix: one shared picker; a field labelled "Folder path" that resolves on Select as well as Enter; refuse with an inline error when it does not resolve; confirm with the absolute path. Suggested: /impeccable harden.
- **[P1] The delete receipt's Undo and Dismiss are composed off the pane.** Why: the receipt is the only recovery path (no Trash view). Cause: PROVEN, `library_notes_canvas.py:872-899` ellipsizes the title to 42 cells inside a 38-column pane in a non-wrapping `Horizontal`. Seen ×3. Fix: width budget derived from the pane, or wrap the buttons to their own row. Suggested: /impeccable layout.
- **[P1] Undo restores the database row and the rail count but the row does not reappear.** B observed once (DB `deleted=0`, count 11, tree still 10, Clear filter no help). I could not re-verify because the button is unreachable at both sizes on my profile. Cause INFERRED: the handler appends to the flat source records and re-syncs, while the tree projection is built from paged branch state. Fix: invalidate the affected branch on restore. Suggested: /impeccable harden.
- **[P1] "Import once" is exiled from its own decision.** Why: the chooser renders one button under the two descriptions; Import once sits 36 rows down beside "Back to Notes". Cause: PROVEN, `_compose_phase` yields only the keep-synced button; `_compose_pinned_actions` yields Import once. Fix: two sibling buttons under their descriptions, Back alone in the pinned bar. Suggested: /impeccable layout.
- **[P1] A brand-new Notes list has no empty state.** Why: the documented copy and the coded copy differ, and neither renders because the tree projection short-circuits the empty branch once `Agent_Lessons` exists (PROVEN, canvas lines 903-905). Fix: render the empty copy above the tree when there are zero notes; explain or hide `Agent_Lessons` until it has content. Suggested: /impeccable onboard.
- **[P1] The list pane is starved to 38 columns at 235 wide.** Why: one geometry decision causes clipped titles, no ages, hidden Move/Remove, the missing "Title" sort option and the clipped receipt; 100x30 reads better than 235x52. Cause INFERRED (observed ×3; stylesheet not traced). Fix: full-width list when nothing is open; 40/60 when a note is open. Suggested: /impeccable layout.
- **[P1] Sort is a dead control.** Why: Newest/Oldest changes the label only; the tree is title-ordered by a pinned decision (`test_placement_title_sort_key_matches_repository_tiebreakers`). Fix: remove the control or make the tree honour it. Suggested: /impeccable distill.
- **[P1] The importer is Obsidian-blind.** Why: no ignore rule for `.obsidian/` or `.trash/` (PROVEN, `note_import_discovery.py` scans every directory); `.json` is a supported type so config files land in "Failed (5)"; frontmatter is never parsed (PROVEN, `_parse_text` takes the first `# ` heading or the stem) so `title:` wins only via the heading and `tags:` produce zero keywords; `{{date}}` becomes a title; wikilinks stay literal; a 2-row CSV is reviewed as one note and imports two. Fix: detect `.obsidian/` and offer one default-on toggle (skip config, trash, templates; frontmatter to title, aliases and keywords; strip the block; rewrite wikilinks to notes created in the same batch; show the effect in the review). Suggested: /impeccable shape.
- **[P2]** `/` inserts itself into the filter it focuses (PROVEN, me + B) · the delete prompt relocates the user, its footer contradicts the focused button, and Tab walks out of it (A) · Escape is refused silently when a save is blocked (B) · a wrong import source cannot be changed without restarting the flow · the review is 71 five-line rows with Skip/Create 130 columns from the path and the receipt lists no skipped files · "Saved" before typing and "Untitled" in the list while the title is typed (on dev after #2531 the refresh is deliberately deferred while the editor has focus, so the row updates on leaving the field) · "Last import" is never offered after Back to Notes (both) · Folder files drops the rail and the source strip and ignores the configured sync folder · duplicate titles indistinguishable, no age rendered (PROVEN: tree rows carry no age label; the age code path belongs to the flat list).
- **[P3]** Burst title+Tab+body interleaves (32062 fixed in part on dev; residual filed) · two contradictory status lines (32063 fixed on dev, not re-verified) · compact editor shows `‹ Notes` where the guide says `‹ Back to list` · grips render as floating ASCII arrows (accessible names exist; pinned) · the wizard Summary points at provider setup three times (dev now offers "Add your first document").

## Retired or re-attributed

| Finding | Verdict |
|---|---|
| B: "Folder files cannot link any folder" (P0) | Retired as stated; re-attributed to the poisoning sequence above |
| A: Ctrl+N dead on the landing (P2) | Pinned decision: `test_library_notes_bindings_are_inactive_outside_notes_workflow`; reported as an accelerator inconsistency |
| A + B: mouse cannot press Info or pinned buttons | Harness limitation (SGR clicks on Textual Buttons); not counted |
| A: grips are unlabelled (Sam) | Accessible names and tooltips exist; a visual disagreement, not a defect |
| B: burst title corruption (P2) | Known; #2531 fixed the recompose cause on dev, residual documented |
| A + B: two status lines | Fixed on dev by #2531 (32063); not re-verified |

## Persona Red Flags

- **Jordan (first-timer):** provider-first hand-off; `Agent_Lessons` unexplained; "Saved" on an empty note; one button for a two-way choice; no Undo after delete; a folder pick that dies silently and then every pick dies.
- **Alex (power user):** `n` vs `ctrl+n`; no ages and 38-column clipping; identical "Reading list" rows; Sort does nothing; `/` corrupts its own query; no bulk skip in a 71-row review.
- **Sam (keyboard, low vision):** focus is shape-visible everywhere checked (strength); Tab leaks out of the delete prompt; Escape refusal is silent; the timeout is silent.
- **Riley (stress):** whitespace-only files, the 120-character name, the emoji title and resize mid-edit all pass; the CSV fan-out and the "Failed" verdict on empty files are honest-copy failures; a wedged session has no recovery but restart.
- **Solo operator:** dry-run and unavailable states are worded well in copy, yet the one failure they will hit (a scan that never ends) says nothing.
- **Researcher/student:** frontmatter tags, wikilinks and templates are exactly the structure a vault is imported for, and all of it is lost or mis-imported.

## Docs vs live

| Guide claim (`notes.md` / `file-notes.md`) | Live |
|---|---|
| "No notes yet. Create one to see it here." | Never renders; code has a different string; neither shows |
| Rows show "title and age" | Title only |
| Rename updates the row "as soon as the note saves" (stamped 2026-09-06) | Updates on leaving the field |
| Status line "N words · saved" | "Saved" / "Saved 04:07"; word count under Info |
| Receipt "offering Undo and Dismiss" | Clipped off the pane |
| Undo "immediately returns its row" | Count returns, row does not (B) |
| "Last import" reopens the receipt | Not offered after Back to Notes |
| Sort strip Newest / Oldest / Title | Title never renders; order unchanged |
| `/` focuses the filter | Also types `/` |
| Folder actions Rename / Move / Remove | Only Rename fits |
| Compact editor `‹ Back to list` | `‹ Notes` |
| Choose folder → "Linked — folder" | True only for a clean first pick; timeout copy never paints |
| "Add another file" for Import once | Not rendered for a folder selection |

## Minor Observations

- Preview shows the body but not the title; `‹ Notes` in Edit and Preview, `‹ Note` in Info; "Saved" printed twice in Info.
- Dates are relative only; no absolute timestamp anywhere for a note.
- "All planned items settled." is opaque copy.
- The picker's own hint ("Enter Open · Select use this folder") never appeared on screen in any capture.
- Empty and whitespace-only files are "Failed … could not be imported safely" rather than "empty".
- The delete receipt survived a whole import journey and was still above the tree afterwards.

## Questions to Consider

- Is "Folder files" a mode of Notes or a different screen? The strip that switches into it disappears once you are there, and the rail goes with it.
- If Import once is the safe default, why is it the only one of the two choices not rendered beside its own description?
- Is the goal to copy a vault or to adopt one? Today the answer is copy, and it costs the researcher persona at the first click.
- Would anyone defend a 38-column list beside 145 empty columns in a design review, or is that just what the container defaulted to?
- What is a "Verified against" stamp worth if nothing re-checks it?

## Improvement opportunities beyond fixes

1. Obsidian-aware import: detect `.obsidian/`, skip config/trash/templates by default, frontmatter to title/aliases/keywords with the block stripped, wikilinks rewritten to notes from the same batch, embeds listed. Researcher, first-timer. L.
2. Review by summary: "45 archive notes · 14 notes · 11 skipped (why)" with expand-on-demand and per-group Skip. Both. M.
3. One shared folder picker: labelled "Folder path", resolves on Select, absolute-path confirmation, last-used start location. Everyone. S.
4. Cancellable scans with a visible budget: a per-directory cancel check, a "Keep waiting / Choose another" prompt after 3 s, and the timeout copy actually painted. Solo operator. M.
5. Give the list the width when nothing is open; full rows with age, folder and a snippet. Everyone. M.
6. A Trash view under the tree ("Recently deleted (N)") so the receipt is not the only safety net. First-timer, stress. M.
7. Backlinks in Info once wikilinks resolve. Researcher. M.
8. Capture from Console: a `/note` command or message action that turns an answer into a note with the conversation as provenance, the reverse of "Use in Console". Researcher, operator. M.
9. Editor chrome strip at the bottom of the body: words, cursor line, save state, and Save / Preview / Keywords / Delete without tabbing. Power user. S.
10. Disambiguate duplicate titles at render time with folder then modified date. Power, stress. S.
11. Folder files as a mode: keep the source strip and rail, offer the configured sync folder as the first suggestion. First-timer. M.
12. A "Start with notes" path from the wizard Summary and Get started for users with no provider. First-timer. S.
