---
target: "Library ▸ Notes — create/edit/link/search/sync/import/export, first-timer and power user (critique #5, ultracode)"
total_score: 16
max_score: 40
na_heuristics: 
p0_count: 4
p1_count: 12
timestamp: 2026-10-03T08-32-24Z
slug: w-chatbook-widgets-library-library-notes-canvas-py
---
Method: dual-agent (A: 7 live persona-journey sub-agents · B: detector/mechanical-matrix + static-sweep sub-agents, isolated), workflows wf_5da07002-b94 + wf_d602b54f-afd; every finding re-reproduced refute-first.

# Library + Library ▸ Notes — Sr Designer / HCI review

Target: origin/dev `2d34cbf80d` (2026-10-02), worktree `.worktrees/notes-library-ux-review`.
Method: dual-assessment impeccable critique — A = 7 live persona journeys (`journeys/`), B = detector + 17-route × 6-size mechanical matrix + static sweeps (`assessment-b/`), isolated from each other; consolidated (`master-findings.md`); every finding independently re-reproduced refute-first (`verify/`) and triaged against backlog/ADRs; improvements from three lenses, ranked (`improvements/ranked.md`). Code maps in `maps/`. 92 agents across two workflows (wf_5da07002-b94, wf_d602b54f-afd).
Harness: isolated tmux profiles (golden: 122 notes / 25 media / 12 conversations / 10 prompts; empty) with a mock OpenAI-compatible LLM; real profile verified untouched.

## Library and Notes: final design/HCI review (origin/dev 2d34cbf80d)


**What was done:**
- 7 live persona journeys, each on its own socket: first-timer Notes, first-timer Library, power-user Notes, power-user Library, accessibility/consistency, stress, researcher loop.
- Assessment B: a 17-route × 6-size capture matrix, footer and focus probes, 9 static code sweeps, and the deterministic detector.
- Every finding was re-reproduced by an independent verifier trying to disprove it (41 single-finding + 17 batch verifiers), then triaged against the backlog and ADRs.

**Result:** 103 findings, none disproved: **7 P0, 24 P1, 52 P2, 20 P3**. 84 have no backlog task, 5 are partly tracked, 14 are already owned by a task.

**Severity rule:** the final severity is what the live re-repro showed. It changed only where triage found an ADR or owner ruling, or a harm the repro didn't weigh, or where the rubric fixes it (data loss or a crash is P0).

**Not covered:**
- Ctrl+digit shortcuts (tmux can't send them).
- What macOS actually does with a `note://` link (I intercepted the call instead).
- Skill trust, because the harness uses a null keyring.
- Answer quality, because the mock LLM only echoes.
- Screen readers and light themes.
- **S-10's full keyboard hang never reproduced naturally** (20 tries); its mechanism was proven by instrumenting focus calls.
- **N-19 was confirmed only by injecting a fault.**
- **S-05's crash is in Study code**, but on the Library → Study path.

### Scores

Each heuristic starts at the median of the journeys that scored it. I lowered one step only where a verified P0/P1 belongs to that heuristic and the scoring journeys didn't weigh it (marked ↓).

| # | Heuristic | Library | Key issue | Notes | Key issue |
|---|---|---|---|---|---|
| 1 | Visibility | 2 | Export freezes silently (L-01); false "no analysis provider" (L-02); "loaded" beside an empty reader (L-16) | 1 | "Saved" / "Sync managed · Ready" while sync is wedged (N-02); "✓ Up to date" after a delete (N-03) |
| 2 | Real-world language | 2 | "workspace workspace-default", "RAG Answer", "invocable" | 2 | "placement", "cutover", nanosecond timestamps |
| 3 | User control | 2 | Esc throws away typed analysis (L-08); the demo installs a paid daily job (L-10) | 2 | Undo is excellent, but Ctrl+Q loses text (N-01) and Recovery loops |
| 4 | Consistency | 1 | "Use in Console" refused from the reader but works from Search (S-01); 6 focus styles | 2 | Arrow keys dead on folder rows; Ctrl+S only in Skills |
| 5 | Error prevention | 1 ↓ | Actions hit the wrong item (L-03); export overwrites files (N-11) | 1 | Quit doesn't save (N-01); a validation veto steals the cursor (N-07) |
| 6 | Recognition | 2 | Notes is not in the nav bar, More menu or palette; read-later items are buried | 2 | No way to make a link; Save hidden at 120 cols |
| 7 | Efficiency | 2 | F6 can't reach Export or Search (S-09); selection doesn't survive a page turn | 1 ↓ | Filter matches whole words only (N-09); list can't scroll (N-04) |
| 8 | Minimalism | 2 | ~19 rows of chrome before a conversation's first message (L-31); Prompts list uses half its pane (L-32) | 2 | At 120x36, 19 chrome rows sit above a 5-row note list |
| 9 | Error recovery | 1 ↓ | "~/ is a dangerous pattern" for a file that exists (L-05); wrong remedy (L-02) | 1 ↓ | "Recovery failed — RuntimeError" loop; false "folder changed elsewhere" (N-24) |
| 10 | Help | 2 | F1 is a bare key list; 8 of 12 guide claims checked are wrong | 2 | F1 shows 3–6 keys and a malformed row |
| | **Total** | **17/40 Poor** | | **16/40 Poor** | |

**Trend.** Library 25 → 17, Notes 25 → 16. This is indicative only: earlier critiques were single-agent passes by different agents.
- **Most of the drop is coverage:** quit, sync edge cases, export and the Console loop were never exercised before.
- **Most earlier issues are fixed.** Two are still present: sync roots named "…before cutover" (N-18, critique #4's N7) and no definitions of terms (L-36).
- **Real regressions since then:** L-11 (task-32954), N-10 (after the Info densification), and a false "changed elsewhere" conflict found inside S-02.
- **Plausible rescore:** if the 7 P0s plus N-04, N-05 and S-01 ship, both surfaces would likely land around 22–24 (Acceptable). That is a guess to confirm with a rescore.

**Design specificity:**
- Specific, not generic. The house grammar (▸/▾ counts, ✓, ⇄, ○ with a reason, █ focus, Undo receipts) is distinctive; the problem is it is applied inconsistently.
- The deterministic scan is **unscannable, not clean**: the detector exited 0 with `[]`, but it only reads web CSS/HTML, not Python or Textual `.tcss`.

**Biggest opportunity:** close the read → note → ask → keep loop. Every joint is broken: S-01 (hand-off refused), S-02 (can't read and take notes together), S-03 (quotes lose their source), S-04 (kept answers lose provenance), S-06 (a second source replaces the first).

### P0 (7)

- **N-01 — Ctrl+Q loses unsaved note text and leaves an empty "Untitled" note.** Typing one character every 1.2 s never saves at all.
  - Fix: add `LibraryScreen.confirm_quit()` that awaits `flush_pending_work()`, with a "Keep editing / Discard and quit" prompt; `prepare_for_quit()` cleans up blank notes; add a ~10 s max-wait to the autosave debounce. NEW.
- **N-02 — One ordinary edit wedges the whole sync root, while the UI says "Ready".** Ctrl+End then typing without Enter triggers it; Recovery loops on "RuntimeError" and both sync directions stop.
  - Fix: compare file bytes with `serialize(note, profile)` at `notes_sync_executor.py:5384` (also :3522 and :5167), and show "Needs attention" on the tree, list and editor. NEW.
- **N-03 — Deleting a synced note leaves the root at "✓ Up to date" with the file still on disk.**
  - Fix: `delete_note` and `restore_note` call `note_changed`; until every write path does, show "Up to date as of HH:MM". Tracked: TASK-32633 (land these two paths first).
- **N-11 — Note and prompt export silently overwrite a file outside Chatbook** (default name in `~`, "Export complete" toast).
  - Fix: an overwrite confirm at the shared FileSave seam (Cancel focused), write via temp file + `os.replace`, remember the folder. NEW. Escalated from P1.
- **L-01 — Media list "Export…" freezes the whole app**, Ctrl+Q included. The canvas's handler waits on a recompose that removes the canvas itself.
  - Fix: hand off through `screen.call_later`, or route the button on the screen like its siblings. NEW (introduced by 480eadbba8).
- **L-02 — Analysis always says "No analysis provider is configured," for every user.** `load_settings()` drops the `[analysis_defaults]` section.
  - Fix: one-line `deepcopy` in `config.py`, plus a guard test; the remedy text is task-28018.
- **S-05 — Study creates a deck, or selecting one, crashes the app** (`TypeError` at `flashcards_handler.py:649`). The dashboard buttons are also clipped, and it says "ready" while it can't generate.
  - Fix: drop `_scope_arguments()` from that call; `height:auto` on the dashboard and card editor; honest local-mode copy. NEW. Escalated from P1.

### P1 (24)

**Notes**
- **N-04** — the list can't scroll at 120 columns or wider. Fix: `overflow-y:auto; height:1fr; min-height:0` on the wide list. NEW.
- **N-05** — the editor header overflows from 120 to ~200 columns (at 119 it fits). Fix: size it from the editor pane's own width, two rows below ~70, drop the 61-column minimum. TASK-32513/32514 cover only the status half.
- **N-06** — any dialog (F1, Move, palette) cancels autosave for good. Fix: re-arm autosave on resume. NEW.
- **N-07** — a validation veto moves the cursor, so text lands in the wrong field. Fix: never move focus on an autosave veto; tidy the field when you leave it. NEW.
- **N-08** — typed `[[links]]` do nothing, and Preview sends `note://` to the OS. Fix: `open_links=False` plus a click handler first, then a `[[` picker. NEW.
- **N-09** — the filter has no prefix matching, word order matters, and keywords aren't searched. Fix: AND-then-prefix matching plus keyword matches. NEW.
- **N-10** — the delete confirmation's buttons are clipped, so you press Delete blind. Fix: `height:auto`, then scroll it into view. NEW.
- **N-13** — creating a note from the palette leaves the tree with no folders, and "Add to folder" is empty. Fix: load the top-level folders. NEW.

**Notes sync**
- **N-15** — reviews for deleted or renamed files can never be resolved, and they block safe changes. Fix: apply the safe subset and show the reason on screen. NEW.
- **N-16** — one latin-1 or mixed-line-ending file blocks the whole folder. Fix: skip that file and name it. TASK-32578 covers only the wording.
- **N-17** — "Add from files" gets stuck on an old receipt or a blank page. Fix: reset to the chooser on open. NEW.
- **N-18** — every sync root is called "Sync folder (name unavailable before cutover)". Fix: show the folder's own name. TASK-32451.

**Library**
- **L-03** — opening an item by link leaves actions aimed at row 1, so Read later, trash and Edit metadata hit the wrong document. Fix: use the canonical id; act on the item that's open in the reader. NEW.
- **L-04** — export bundles drop keywords for media, notes and conversations. Fix: fetch keywords in batch and re-link them on import. NEW.
- **L-05** — `~/` paths are called "dangerous" and then "can't find that path". Fix: expand `~` at the pre-check. NEW.
- **L-06** — RAG Answer ignores the Console's provider and model (even with Anthropic it bills OpenAI's default) and names the model only after the paid call. Fix: resolve from `[chat_defaults]` and name the model before Run. NEW.
- **L-08** — Esc throws away a typed analysis. Fix: on Esc with unsaved changes, ask Save or Discard (ADR-055 pattern D). NEW.
- **L-10** — "Try report demo" sets up a daily paid brief plus hourly fetches without warning. Fix: rename it "Set up a daily brief…" and confirm first. NEW.
- **L-18** — the export screen says "copies full media files" but writes text only, and a same-day default name overwrites. Fix: drop the false quality choice; add " (2)" to the name. TASK-32381/32382. Raised from P2.

**Cross-destination**
- **S-01** — media "Use in Console" is always refused with a raw workspace id, while Search stages the same item. Fix: link to the workspace on use (as notes and conversations already do) and show the display name. NEW.
- **S-02** — the reader and a note are never on screen together, and switching discards the reading position and closes a new note. Fix: keep both states; add a "Take note" key in Media. NEW.
- **S-10** — keyboard focus can land on a widget being removed, locking all keys. Fix: `set_focus` refuses detached widgets, plus an app-level guard. NEW.
- **S-17** — a blocked nav-bar switch is silent, highlights the wrong tab, and swallows the retry click. Fix: restore the highlight and show the reason. NEW. Escalated.
- **S-19** — after a send, Console contradicts itself about staged sources, and follow-ups go out ungrounded. Fix: append to TASK-33620.7 / 33621.20.

### P2 headlines (52)

**Notes:**
- **N-12** — cancelling export says "Export failed" and focus jumps to Delete.
- **N-14** — selection can only export, and "Select all" miscounts.
- **N-19** — "No writes yet." shown when the history read failed.
- **N-20** — exported YAML front matter is invalid for titles with a colon.
- **N-21** — 8 Tabs from filter to result.
- **N-22** — arrow keys skip folder rows.
- **N-23** — folder picker misses nested folders.
- **N-24** — Move fails at random with a false "folder changed elsewhere".
- **N-25** — the leave-veto message names a button that isn't there.
- **N-26** — a new note shows stale backlinks.
- **N-27** — "View imported" lands on a collapsed folder.
- **N-28** — moving into a collapsed folder hides its notes.
- **N-29** — no permanent delete.
- **N-30** — sort chooser opens with focus in the filter.
- **N-32** — F6 in the note body selects a line instead of moving on.
- **N-35** — raw nanosecond timestamps and class names in sync copy.
- **N-36** — Session Git shows no diff.
- **N-37** — unlinked Folder files is mostly blank, and at 120 cols the side panel vanishes.

**Library:**
- **L-07** — Add analysis loses focus, so typed letters fire commands.
- **L-09** — `c` resumes a conversation the filter has hidden.
- **L-11** — the skill switch shows no visible OFF state.
- **L-12** — Media and Skills open with an empty reader at 64–95 columns.
- **L-13** — a plain question gets "No evidence".
- **L-14** — Find counts lines, not matches.
- **L-16** — a "loaded" row sits beside an empty reader.
- **L-17** — Select evidence jumps focus to a source checkbox.
- **L-19** — selection is limited to one page.
- **L-20** — imported items are titled by filename.
- **L-21** — HTML import keeps site navigation.
- **L-22** — "Open original" does nothing.
- **L-23** — source toggles push the answer below the fold.
- **L-26** — `{{notes}}` silently becomes `{notes}`.
- **L-27** — replaying history spends money.
- **L-28** — Customize hides the built-in skill.
- **L-30** — no way to cancel a batch import.
- **L-31** — conversation reader chrome.
- **L-32** — Prompts list uses half the pane.
- **L-33** — a hidden model download, plus a Hugging Face warning drawn over the nav bar.
- **L-34** — "keep for later" lives in three places.

**Cross-destination:**
- **S-03** — no quote-to-note with a source link.
- **S-04** — kept answers have no provenance.
- **S-06** — a second staged source replaces the first.
- **S-07** — focus is lost after F1.
- **S-08** — fast keys after a filter submit fire `i` / `n`.
- **S-09** — F6 can't reach Export or Search.
- **S-11** — Escape semantics differ per screen.
- **S-13** — arrival focus lands on "⌃1 Home".
- **S-15** — footer and F1 disagree.
- **S-16** — labels cut mid-word at 80 columns.
- **S-18** — a deleted staged note still shows "Ready".
- **S-20** — Console's Search Library modal.
- **S-21** — Import… only imports media.

### Roadmap

**Fix first (small, one place each):**
- N-01 + N-06: the quit and dialog autosave hooks.
- L-02: the one-line `config.py` fix.
- L-01: `call_later` hand-off.
- N-02: compare bytes, not text.
- N-11: overwrite confirm.
- S-05: the crash.
- CSS one-liners: N-04, N-10, L-32.

**Quick wins:**

| Item | Findings |
|---|---|
| Link media to the workspace on use | S-01 (R-02) |
| Name sync roots; honest history copy | N-18, N-19, N-35 (R-03) |
| Name the model before a paid Ask; replay in its own mode | L-06, L-27, L-24 (R-04) |
| Keyboard fixes | N-32 (F6 priority), N-22 (folder rows), N-30, S-27, S-17 |
| Load folders on create; reset facts on a new note | N-13, N-26 |
| Honest demo label and confirm | L-10 |
| F1 "How it works" | S-15, L-36 (R-05) |
| Reach Notes from the palette | R-06 |

**Next (medium):**
- N-05: editor header sized by pane width.
- A focus lifecycle that tells a real arrival from a closed dialog: S-07, S-13, S-10, S-08.
- R-08 input-safety contract: N-07, N-25, N-12, L-05, L-07, L-08, L-30.
- R-09 accumulating staged sources: S-06, S-18, S-19, S-20. **Needs owner approval** (Console Inspector).
- R-10 one sync-health view: N-03, N-15, N-16, N-17.
- R-12 export that keeps provenance: L-04, N-20, L-18.
- R-13 one query grammar: N-09, L-13.
- R-14 link-native notes: N-08.
- R-15 capture with provenance: S-03, S-04. **Needs owner approval.**

**Strategic (needs an ADR change or owner ruling):**
- R-23 the Default workspace means "everything local" (revisits the 32107 decision).
- R-28 a reading desk: needs ADR-086/084 amendments and owner approval.
- R-27 one verb/key grammar: ADR-031.
- R-29 selection that survives paging, with real bulk actions: ADR-067/055.
- R-24 one answering engine.

### Persona red flags

- **Jordan (first-timer):** gave up on linking; deleted blind at 120 cols; told an existing file is "dangerous"; Read later saved the wrong item; Use in Console refused; Export froze; lost text on quit.
- **Alex (power user, Notes):** sync wedged while it said "Ready"; "Up to date" after a delete; partial and keyword search return nothing; links are dead; the list won't scroll; Ctrl+Q ate a sentence.
- **Morgan (power user, Library):** billed an unchosen model; one history click spent money; the demo signed up for a paid daily job; analysis is unusable; typed analysis lost twice; exports drop keywords.
- **Sam (keyboard-only, low vision):** invisible focus on clipped controls; focus lost after every F1; six focus styles; active modes unmarked; cancel reported as failure with focus on Delete; one full keyboard lockout at 80x24.
- **Riley (stress tester):** quit loses text and leaves orphan notes; a dialog kills autosave; a veto steals the cursor; export overwrites a real file; one latin-1 file blocks a sync folder; deleted or renamed files can never be resolved.
- **Priya (researcher):** reader and note never share the screen; quotes lose their source; kept answers have UUID keywords and no sources; Study says "ready", then shows nothing or crashes; exported YAML is broken and tags are gone.

### Cognitive load

- **Notes:** fails 5 of 8 checks (high). It fails single focus, chunking (9 toolbar controls), visual hierarchy, minimal choices, and working memory (you need a UUID to link; you can't see reader and note together).
- **Library:** fails 4 of 8 (high). It fails single focus, visual hierarchy, minimal choices, and working memory (three places for "later"; the search mode changes silently).

### Appendix

**Disproved findings: none.** Verification did correct details inside surviving findings:
- **S-18:** the deleted note's text is not sent; it is silently dropped.
- **L-09:** Archive does nothing at all, rather than acting on the wrong item.
- **L-12:** Conversations is not empty; it auto-loads a transcript.
- **S-16:** the nav-bar cuts are invisible by design.
- **S-25:** the "stale Next: Start typing" claim is wrong.
- **N-24:** not specific to Unfiled notes.
- **N-33:** the missing Ctrl+S in Notes is ADR-031 by design; Skills is the outlier.
- **N-05:** the proposed fix would read the wrong pane's width.

**Conflicting severities between journeys, resolved:**

| Finding | Journeys said | Final |
|---|---|---|
| N-04 | P0 vs P1 | P1 |
| N-05 | P3 to P1 | P1 |
| N-06 | P0 vs P1 | P1 (the loss belongs to N-01) |
| N-18 | P3 to P1 | P1 |
| N-17 | P2 vs P1 | P1 |
| S-01 | P2 vs P1 | P1 |
| L-06 | P2 vs P1 | P1 |

- F6 (N-05 vs N-32): both claims are true from different starting fields.
- L-01's freeze happens only from the Media list toolbar, not the other export paths.

## Appendix — all verified findings (103)

Source of truth: `findings.json`. Evidence paths are relative to this directory. `tracked_by` = existing backlog task(s); NEW = untracked.

| ID | Sev | Surface | Finding | Tracked | Fix |
|---|---|---|---|---|---|
| L-01 | P0 | Export | Media list 'Export…' freezes the whole app (no input, no repaint, Ctrl+Q dead) | NEW | Make LibraryMediaCanvas's export forwarder synchronous and hand off via self.screen.call_later, or route #library-media-export on the screen like its siblings (library_screen.py:29708). Guard _apply_library_open_item_surface against awaiting recompose from a descendant. Pilot test: press export, then Escape, and the app responds. |
| L-02 | P0 | Media | Analysis always says 'No analysis provider is configured', even when [analysis_defaults] names a ready provider | task-28018 | Add 'analysis_defaults': copy.deepcopy(toml_config_data.get('analysis_defaults', {})) to load_settings (config.py ~2605). Add a guard test that every table read through app_config.get exists. Fix the false remedy 'Settings ▸ Providers & Models' (task-28018). |
| N-01 | P0 | Notes | Ctrl+Q quits without flushing: unsaved note text is lost, and a new note becomes an empty 'Untitled' row | NEW | Add LibraryScreen.confirm_quit() that awaits flush_pending_work() (library_screen.py:11109). On a veto or failure, ask 'Quit and discard unsaved changes to "<title>"? Keep editing · Discard and quit', with Keep editing focused. prepare_for_quit() runs the blank-note GC. Add a ~10 s max-wait to the 2 s autosave debounce. Arch test: any Screen that defines flush_pending_work must define confirm_quit |
| N-02 | P0 | Notes-sync | One ordinary edit to a synced note wedges the whole sync root (postcondition_failed); Recovery loops on 'RuntimeError' while Notes says 'Saved' / 'Sync managed · Ready' | NEW | In _classify, compare file bytes with serialize(note.content, profile) (notes_sync_executor.py:5384; also :3522 and :5167). Recovery then self-heals roots that are already stuck. Add postcondition_failed to _CHECK_FAILURE_ROW with plain copy and a real next action. While an op is incomplete, the tree row, list scope and editor location line read 'Needs attention', not 'Ready'/'Saved'. |
| N-03 | P0 | Notes-sync | Deleting a synced note leaves the root at '✓ Up to date' while the file still exists | TASK-32633 | Within 32633, raise priority and land delete_note/restore_note first: call note_changed(note_id), as note_session_port.py:190 does. Until all 17 write paths signal, render '✓ Up to date as of HH:MM'. The delete confirm for a synced note says the file stays on disk. |
| N-11 | P0 | Export | Note and prompt Export silently overwrite an existing file outside Chatbook (title-derived name, starting at ~) | NEW | Add an overwrite confirm at the shared FileSave seam ('Replace "x.md" in ~/exp? Cancel · Replace', Cancel focused) for the note, prompt, artifact and collections exports. Write via a temp file plus os.replace. Remember the last export directory. Toast the full path. |
| S-05 | P0 | Cross-destination | Study hand-off dead-ends: a whole-Library 'ready' snapshot, clipped Dashboard/Flashcards controls, server-only generation, and creating or selecting a deck crashes the app | NEW | Crash first: drop **self._scope_arguments() from the list_flashcards call (UI/Study_Modules/flashcards_handler.py:649), or add those params to StudyScopeService.list_flashcards. Then set height:auto on the dashboard columns row, .study-dashboard-column and .card-editor. Use honest local-mode handoff copy ('Generating a pack from sources needs a tldw server'). Show the true scope instead of '… and  |
| L-03 | P1 | Media | Opening an item by link leaves the list selection on row 1, so Read later, trash and Edit metadata act on a different document | NEW | Canonicalise record_id to 'local:media:N' in the media branch of _open_library_item_by_id (library_screen.py:34798). Reader-scoped actions act on reader_session.loaded_id. The trash confirm names the item. |
| L-04 | P1 | Export | Export bundles silently drop keywords for media, notes and conversations | NEW | Creator: fetch keywords with fetch_keywords_for_media_batch, get_keywords_for_notes_batch and get_keywords_for_conversations, and write them into metadata and ContentItem.tags. Importer: re-link note and conversation keywords (the ADR-057 precedent). Add a round-trip test per type. |
| L-05 | P1 | Ingest | Import rejects '~/' paths as a 'dangerous pattern', then says 'Can't find that path' for a file that exists (and a fast Enter imports anyway) | NEW | Expand '~' at the preflight boundary (ingest_preflight.py:421). The submit path already expands it. Enter waits for the current preflight. Show one plain message per failure. |
| L-06 | P1 | Search-RAG | RAG Answer ignores the Console provider and model, calls OpenAI's handler default (gpt-5.6-terra), and names the model only after the paid call | NEW | Resolve provider and model from [chat_defaults] the way briefings do (resolve_remembered_provider_model), and pass the model through. The pre-run line reads 'To OpenAI · gpt-4.1-mini: question + evidence'. Apply the same fix to the scheduled-automation path. |
| L-08 | P1 | Media | Escape in the analysis or metadata editor discards typed text without asking | NEW | Apply ADR-055 Pattern D: record a baseline when the form opens. A dirty Esc keeps the form, and the footer reads 'esc save or discard first'. While dirty, Cancel becomes 'Discard changes'. Same for the metadata form. |
| L-10 | P1 | Artifacts | 'Try report demo' creates a persistent live-RSS watchlist with a daily paid brief and hourly fetches, disclosed only afterwards | NEW | Relabel it 'Set up a daily brief…' and add an inline confirm naming the feeds, the 24 h brief, hourly source checks, the provider and model, and how to remove it. Put the CTA under the empty-state sentence. Share the copy with artifacts_screen.py (the ADR-079 consequence copy was dropped in the port). |
| L-18 | P1 | Export | The Export canvas promises 'copies full media files into the zip' but writes text only; a same-day export overwrites by default; the size is never reported | TASK-32381, TASK-32382 | Remove the inert quality chooser and show one fixed line: 'Text and metadata only — original media files are not included'. Add a README that maps media_<id>.txt to titles (keep the names; the importer relies on them). The default name gets ' (2)' when it already exists. The receipt shows count and size. |
| N-04 | P1 | Notes | At ≥120 columns the Notes tree cannot scroll: notes, 'Load more notes' and 'Recently deleted' below the fold are unreachable, and focus moves onto rows that are never shown | NEW | Give the wide #library-notes-list 'height:1fr; min-height:0; overflow-y:auto; overflow-x:hidden' (css/features/_library_panels.tcss:543-548) and regenerate the bundle with css/build_css.py. Pilot test at 120x36 and 160x45: max_scroll_y > 0, and the 30th focused row is visible. |
| N-05 | P1 | Notes | Note editor header overflows from 120 to ~200 cols: Save, 'Use in Console' and 'Discard new note' are clipped or missing, save state is a 1-column strip, Tab/F6 land on hidden controls | TASK-32513, TASK-32514 | Choose the header shape from the editor pane's own width, not the 120-col shell breakpoint. Below ~70 pane cols use two rows: [Edit · Preview · Info] and [Save · Use in Console · Discard new note]. Drop the 61-cell min-width. The status gets its own row, or moves to the chrome strip (32513/32514). Never truncate 'Use in Console'. F6 skips clipped targets. |
| N-06 | P1 | Notes | Opening any modal (F1, Move note, command palette) cancels the pending autosave and never re-arms it; the status keeps promising automatic save | NEW | In on_screen_resume, call _schedule_library_note_autosave() when the note session is dirty. Show 'changes save automatically' only while a save is actually scheduled. |
| N-07 | P1 | Notes | An autosave validation veto (trailing space in title, duplicate keyword) moves focus mid-typing, so the next words land in the wrong field | NEW | Pass an explicit/autosave flag into _apply_library_note_save_outcome (library_notes_controller.py:3945). On an autosave veto, mark the field inline and never move focus or switch to Info. On blur, strip title edge whitespace and dedupe keywords as a visible draft edit (ADR-027-compatible). While a veto shows, drop the 'save automatically' suffix. |
| N-08 | P1 | Notes | Typed [[wikilinks]] are dead text and are not counted as backlinks; Preview sends note:// links to the OS URL handler | NEW | Ship first: Preview Markdown with open_links=False plus a LinkClicked handler, so note:// opens in the app (library_notes_canvas.py:2909). Then: a '[[' title picker that inserts the stored form, 'Copy link to this note' in Info, and a note_links edge for a bare [[Title]] that resolves to exactly one note (within the task-32263 ruling). |
| N-09 | P1 | Notes | Notes filter matches only whole-word phrases in title and body: no prefix, word order matters, keywords are not searchable | NEW | At note_folder_repository.py:808, use and_then_prefix (AND the tokens, prefix-match the last); treat input as a phrase only when it is quoted. UNION keyword matches through note_keywords. The zero-result copy names what is searched. Plan-pin any new index (CLAUDE.md gotcha 1). |
| N-10 | P1 | Notes | The inline delete confirm is clipped inside Info: Cancel and Delete are pressed blind | NEW | Add the base rule '#library-note-delete-confirmation { height:auto }' (it is currently a 1fr Vertical with overflow hidden). Show the block, then call_after_refresh(scroll_visible), then focus Cancel. Pilot test at 120x36 and 80x24. |
| N-13 | P1 | Notes | Arriving through Create (palette 'New Note' or rail New note) shows the Notes tree without its folders, and Add to folder lists nothing | NEW | In _create_library_note, request the root (None,'folders') branch when it is missing, before the locate. Never paint label-less rows: show 'Loading folders…'. Regression test: palette Create > Blank > Esc, then assert folders are present and the Add-to-folder choices are not empty. |
| N-15 | P1 | Notes-sync | Sync review rows for files deleted or renamed can never be resolved, and they block every safe change in the root | NEW | Let Apply reviewed apply the safe subset (_apply_blocker; runtime _blocked_plan_status). Show the blocker as visible text, not a hover-only tooltip. Enable 'Disconnect item'. Then build 'Restore missing side' and 'Delete counterpart' (move to trash, never unlink), and treat a rename as a move. |
| N-16 | P1 | Notes-sync | One non-UTF-8 or mixed-newline file blocks the whole sync folder with 'Check failed — NotesSyncRootRefused' | TASK-32578 | Record per-file gate refusals as skip rows that name the file, instead of raising for the whole root (notes_sync_runtime.py:930-946). Keep the _CHECK_FAILURE_ROW and _CHECK_REFUSAL_COPY key sets equal, pinned by a test. |
| N-17 | P1 | Notes-sync | 'Add from files…' sticks on a stale lasting-sync phase (old receipt, configure form or a blank page), hiding Import once until restart | NEW | Add begin_add_from_files(), which resets the phase to 'choose' unless a check or activation is running. abandon_setup ends in 'choose'. The canvas renders the chooser for any phase it does not handle. The receipt bar gets 'Manage sync folders'. |
| N-18 | P1 | Notes-sync | Every synced folder is titled 'Sync folder (name unavailable before cutover)'; roots and receipts cannot be told apart | TASK-32451 | Carry display_name in NotesSyncRootRuntimeSnapshot and use it as the row title. Fall back to 'Synced folder N', never 'cutover'. Prefix each receipt with its root name. Order rows by creation, not by uuid. |
| S-01 | P1 | Cross-destination | Media 'Use in Console' is always refused ('Copy or link this media into workspace workspace-default…') with no control that fixes it, while Search evidence stages the same item | NEW | Port link-on-use to _open_selected_media_handoff (library_screen.py:35519): link the item to the active workspace, stage it, and show 'Linked to Local Default · staged in Console · Undo link'. Show the display name, not the raw id (eligibility.py:79). Route Search 'u' through the same gate. |
| S-02 | P1 | Cross-destination | Reader and note can never be on screen together; a rail switch throws away the reading position and closes a newly created note; Media has no 'take note' key | NEW | (1) Keep a saved note session across rail switches (drop the session_blank_id gate once content is saved). (2) Keep the open media id, tab and scroll across rail presses, and focus its row. (3) Fix the retained-editor render that shows a false 'changed elsewhere' conflict. (4) Add 'Take note' (n) to the Media reader. A split view needs an ADR-086 amendment (Q4). |
| S-10 | P1 | Library-shell | Keyboard input can stop entirely after a destination switch (focus lands on a widget being removed) until a mouse click | NEW | BaseAppScreen.set_focus refuses detached or pruning widgets. The list-entry retry skips pruning candidates and counts only attached focus as landed. An app-level guard refocuses when screen.focused is detached. Add a race Pilot test. |
| S-17 | P1 | Cross-destination | A vetoed nav-bar switch is silent, highlights the wrong destination, and swallows the retry click | NEW | In the app_navigation.py veto branch, call restore_active(current route) and notify the screen's veto message. LibraryScreen.flush_pending_work notifies validation vetoes with the field and reason. |
| S-19 | P1 | Cross-destination | After a send, Console contradicts itself about staged sources, and follow-ups (even in a new tab) go out ungrounded with a log-only error | TASK-33620.7, TASK-33621.20 | Append the Library-origin repro to 33620.7 and 33621.20. Release the consumed launch on the first-send path. Give the strip and the counts one source of truth. Log type(exc).__name__ (no traceback) at console_chat_controller.py:24676. |
| L-07 | P2 | Media | 'Add analysis' puts focus on the first Items row, so typed text is lost and 'i' or 's' fire list commands | NEW | Call _after_library_media_viewer_sync('#library-media-analysis-edit-text'), or '#library-media-edit-title' for Edit metadata. Cancel and Save return focus to the button that opened the form. |
| L-09 | P2 | Conversations | When a filter hides the loaded conversation, 'c' resumes it, Use as source links it and then refuses, and Archive does nothing | NEW | When a settled filter drops the loaded conversation, load the first match (Media's filter_select_first), or label the Reader 'Not in current results' and disable c, Resume, Use as source and Archive. Write the workspace link only after staging succeeds. |
| L-11 | P2 | Skills | The built-in skill 'Enabled' switch shows OFF as an empty box (1.22:1) with no text change | NEW | Replace the Switch with a Button labelled with library_toggle_label: 'Enabled: ✓ on ⇄ off'. Drop the ✓ on list rows marked Disabled. |
| L-12 | P2 | Library-shell | At 64–95 columns, Media and Skills open with the list collapsed beside an empty Reader | TASK-32304, TASK-31568 | Remove the <64 cap on list_first_when_empty in resolve_adaptive_reader_layout. Set the flag on the Skills, Conversations and Collections profiles. Delete Skills' 82-col items floor. Re-pin the 80x24 tests. |
| L-13 | P2 | Search-RAG | Search mode returns 'No evidence matched' for a plain question the documents answer; the rail box silently switches the mode to Search | NEW | For question-shaped queries, use zero-result copy that says Search matches every word, with an 'Ask as RAG Answer' button that does not auto-run. Use a placeholder per mode. When the rail forces Search, say so on the run's status line. |
| L-14 | P2 | Media | In-document Find counts lines, not matches, and marks nothing in the default Rendered view | TASK-32384, TASK-28021 | Return (line, col) per occurrence. Highlight and step through every occurrence. While a query is active, show Raw with the note 'Showing raw text while Find is active'. |
| L-16 | P2 | Media | Returning to Media via the rail shows a row marked 'loaded' beside an empty Reader | NEW | Rebuild reader_session in the rail-switch reset (library_screen.py:22391) after capturing reading progress. |
| L-17 | P2 | Search-RAG | 'Select evidence' remounts the cards, so focus lands on a Sources checkbox and the panel jumps to the top | NEW | Update the selection label and class in place, or capture and restore the focus id and scroll offset for any focused descendant of a card. |
| L-19 | P2 | Media | Media selection is cleared by any page turn, and there is no bulk keyword action | NEW | Amend ADR-067 to give Media the Prompts cross-page basket ('N selected · M on this page'). Add 'Keywords…' to the select toolbar, under the overflow at <110 cols. |
| L-20 | P2 | Ingest | Imports are titled by filename stem; PDF Title/Author, HTML <title> and Markdown H1 are ignored | NEW | Stop pre-filling the title with file_path.stem (local_file_ingestion.py:1066). Read result['metadata'] for title and author (:1670). Use <title>/og:title/h1 for HTML and front matter or the first H1 for Markdown, with the stem last. |
| L-21 | P2 | Ingest | Local HTML import keeps site navigation, duplicates the heading and flattens tables into RAG chunks | NEW | Use shared trafilatura extraction (include_tables=True, markdown) for both local and URL HTML. The BeautifulSoup fallback strips head, nav and footer, and joins table cells with ' \| '. |
| L-22 | P2 | Media | Reader 'Open original' and 'Open manager' do nothing for a local import | NEW | Offer 'Show in Finder' / 'Show in folder' only for an existing file:// source, and reveal rather than open it. Hide the button for local:// sources. Remove 'Open manager', which navigates to the screen it is already on. |
| L-23 | P2 | Search-RAG | Search/RAG Sources toggles take 3 rows each and push the answer below the fold at 100x30 | NEW | Show the Sources toggles on one row in a reflowing grid. When results arrive, reveal the Answer heading, or keep the focused query at the top (the TASK-32751 contract). |
| L-26 | P2 | Prompts | Prompt 'Use in Console' turns {{notes}} into a literal {notes} with no warning, and leaves Instructions out (off by default) under a different name | NEW | Show an authoring hint when '{{name}}' appears. Add the dialog status 'No variables — {{notes}} will be inserted as {notes}'. Add an 'Instructions not applied' line and receipt. Title the dialog 'Use prompt in Console' when there are no variables. Do not collapse prompt inserts into the pasted-text chip. |
| L-27 | P2 | Search-RAG | Replaying a Recent search under RAG Answer silently makes a paid call | NEW | In RAG mode a history row only fills the query and focuses Run ('Press Run to ask <model> again'). Later, store the mode per entry and mark paid ones 'RAG · paid'. |
| L-28 | P2 | Skills | Skill 'Customize' hides the built-in and leaves a copy the agent cannot run, without warning beforehand | NEW | Add an inline, trust-aware consequence line under Customize. After copying, open the copy in Trust or Edit. An overriding copy shows 'Replaces the built-in · Reset to built-in…'. Clear the stale › marker. |
| L-30 | P2 | Ingest | Batch import has no cancel, and Enter imports while the footer still says 'check this path' | NEW | Resync the footer on every path keystroke. The first Enter checks and shows the forecast; the second imports. Add 'Cancel remaining (N)' to local batch rows. |
| L-31 | P2 | Conversations | The Conversations reader stacks ~19 rows of chrome above the first message; at 80x24 no message is visible | NEW | Open Find on demand (ctrl+f). Use a one-line status. Put the actions in a wrapping grid. Cut the link-on-use reason to one line. Drop the 'Conversation reader' heading. Target: the first message at or above pane row 8 at 120x36. |
| L-32 | P2 | Prompts | The Prompts list gets half the pane; the other half is blank | NEW | Add classes='library-source-pager' to #library-prompts-pager (library_prompts_canvas.py:1082), as Skills and Media do. |
| L-33 | P2 | Ingest | The first import downloads an embedding model without saying so, and a Hugging Face warning is painted over the nav bar | NEW | In protect_file_descriptors, bind stdout/stderr to an fd-backed devnull when Textual's capture has no fd (Utils/fd_protection.py:151). Use TextualHandler(stderr=False) once the app is mounted. Show 'Preparing search index — one-time model download' on the queue row. |
| L-34 | P2 | Collections | 'Keep for later' is split across Media Read later, Collections Reading and Home; Home's count lands on the wrong store; the Quick Capture note box has no label | NEW | Give the box the placeholder 'Note (optional)'. The Read later confirmation names Media ▸ Sets ▸ Review read-later (N), and that button shows the count. Fix Home's deep link (task-32910). Add a cross-reference line to the Collections empty state. |
| N-12 | P2 | Notes | Cancelling a note export is reported as 'Export failed', and focus lands on Delete | NEW | Treat selected_path None as cancellation: no failure copy and one quiet signal. After any outcome, refocus the button that started the operation. transfer_running disables the focused button, so focus currently jumps to Delete, and that also happens after a successful export or a Copy. |
| N-14 | P2 | Notes | Notes select mode can only Export; 'Select all N shown' miscounts; selecting costs two keys per row (Space does nothing) | TASK-32635 | Compute the Select-all label from the rendered ids ('Select all 20 loaded'). Space toggles the focused row (copy Media's priority binding). Add an 'Actions…' chooser: Move to folder, Add/Remove keyword, Delete with one undo receipt. |
| N-19 | P2 | Notes-sync | Sync Receipts says 'No writes yet.' when the history could not be read | NEW | Count failed write_receipts reads and show 'Couldn't read sync history for N folder(s) · Retry'. Use the empty copy only when every read succeeds. Iterate _all_roots, not just one page. Refresh when the runtime turns active. |
| N-20 | P2 | Export | Exported note front matter is invalid YAML for ordinary titles (':', '*', a leading quote; '#' silently becomes null) | NEW | Add one render_front_matter() helper using yaml.safe_dump (lazy import) for chatbook_creator.py:1463 and library_notes_state.py:914. Emit tags as lists. Round-trip test through yaml.safe_load and the app's own _split_frontmatter. |
| N-21 | P2 | Notes | No fast path from a filter query to a note, and no multi-hop note history | NEW | Enter on an unchanged filter opens the first result; Down in the filter focuses the first row. Following a backlink shows a one-level '‹ <origin note>' back cue. Defer an 'Open note…' palette provider. Do not use the 'o' key (it already means open evidence). |
| N-22 | P2 | Notes | Arrow keys skip folder rows, and there is no expand/collapse key in the Notes tree | NEW | Add 'library-notes-folder-row' and 'library-notes-tree-pager' to _LIBRARY_LIST_ROW_CLASSES (screen_constants.py:434). On a focused folder row, Right expands and Left collapses or jumps to the parent. |
| N-23 | P2 | Notes | Folder pickers list only folders whose branches were loaded in this visit; nested targets are missing on a fresh visit | NEW | Make the picker a type-to-filter list over all active folders (20 at a time, exact total). Titles name the item: 'Move "<title>" to…'. |
| N-24 | P2 | Notes | 'Move note' fails intermittently with a false 'That folder changed elsewhere — refresh and try aga…', with no Refresh control | NEW | Diagnose the spurious FolderConflictError on the move path. Hypothesis: SQLite BUSY/LOCKED on a borrowed stale transaction is mapped to 'conflict'. Give conflict reasons their own copy plus a real Refresh button. Clear the selection after a delete. Wrap the notice instead of truncating it. |
| N-25 | P2 | Notes | The leave-veto toast names a 'Discard new note' button that is never shown, and blames the title for every veto | NEW | Build the toast from save_outcome.veto (field and reason). Mention 'Discard new note' only while that button is visible, which is effectively never at veto time. |
| N-26 | P2 | Notes | Info 'Linked from' never resolves on a new note, or shows the previous note's backlinks | NEW | Add one shared _reset_library_note_facts() used on create as well as on open. A fresh note sets backlinks=() and 'ready', and resets its location. |
| N-27 | P2 | Notes-import | 'View 10 imported notes' lands on the list with the import folder collapsed | NEW | Locate the first imported note with _locate_library_notes_tree_target(focus=True). Do this after the N-28 fix. |
| N-28 | P2 | Notes | Moving a folder into a collapsed folder hides the target's own notes until it is collapsed and re-expanded | NEW | The locator loads the placements slice for every folder it newly expands, through a shared _ensure_library_notes_folder_slices(). |
| N-29 | P2 | Notes | Notes 'Recently deleted' can never delete permanently, and holds only the newest 20 with no pager | NEW | Add 'Delete forever' (key x; ADR-055 Pattern B confirm naming the title), backed by a hard-delete seam that purges FTS, keywords, memberships, versions and sync_log. Refuse, or warn, for synced notes. Add an ADR-067 pager. |
| N-30 | P2 | Notes | The Notes sort chooser opens with focus left in the Filter field | NEW | In handle_library_notes_sort, use then= to focus the active '✓' choice (_focus_library_choice_strip_active). Return focus to #library-notes-sort on close. |
| N-32 | P2 | Notes | F6 in the note body selects the current line instead of moving to the next pane | NEW | Add a priority Binding('f6','focus_next_workbench_pane') to LibraryScreen.BINDINGS beside shift+f6 (library_screen.py:1022). Not app-level: ADR-031 keeps the global F6 non-priority. |
| N-35 | P2 | Notes-sync | Sync comparison shows raw nanosecond and ISO timestamps, exception class names and stale status lines | TASK-32578 | Show local 'YYYY-MM-DD HH:MM' on both sides and mark the newer one. Replace the type-name fallback with a plain sentence plus a next action. Gate the history-disabled line on the root's activation state. |
| N-36 | P2 | Notes-folder-files | Folder files' Session Git shows status but never a diff, even at Confirm commit | NEW | Add a read-only 'View diff' per row (git diff --no-ext-diff --no-textconv, --cached for staged), bounded with the conflict-compare limits. Add a staged-diff disclosure to Review commit. |
| N-37 | P2 | Notes-folder-files | Folder files with no folder linked is 61–90% blank, the rail jumps 3–4 rows, and at 120x36 the rail is not rendered at all | NEW | Keep the work pane mounted before a folder is linked, with a centred 'No folder linked · Choose folder…' empty state. Move the authority and root rows into the work column so the rail stays put. Hide 'Review recovered pairing…' until a pairing exists. |
| S-03 | P2 | Cross-destination | No way to quote a passage into a note with a link back to its source | NEW | Add 'Quote to note…' on highlight cards and a Reader selection key, inserting '> quote — [title](media://id)'. Render media:// in Preview. Show 'Notes citing this item' in media Info. Do not rebind Ctrl+C. |
| S-04 | P2 | Cross-destination | Kept AI answers lose their provenance (no question, source or model; UUID keywords nobody can search); RAG Answers cannot be saved | NEW | Add a provenance footer to captured answers (question, conversation title, date, model, Sources with links). Info shows 'From conversation: <title> · Open'. Add 'Save answer as note' to RAG Answer. Make the provenance keywords searchable. |
| S-06 | P2 | Cross-destination | A second 'Use in Console' silently replaces the first staged note | NEW | Make stage_console_staged_evidence additive (merge, dedupe, renumber, cap 20), built on 33620.7's per-session keying. Status: 'Added to Console · 2 sources staged'. Add 'c use selected in Console' to Notes select mode. |
| S-07 | P2 | Library-shell | Closing F1 or any modal drops keyboard focus on the landing, the Notes navigator (including a just-shown Undo), Conversations and Skills | NEW | In on_screen_suspend, record whether an overlay was pushed. On that resume, skip _refresh_library_visit_surfaces and the Notes canvas recompose. The landing restore searches the landing canvas. Add Pilot tests that F1/Esc keeps focus on each surface. |
| S-08 | P2 | Library-shell | Keys typed within ~0.15 s of submitting the Notes filter fire screen shortcuts: 'i' opens Import media, 'n' creates a stray note | NEW | Keep the filter Input mounted and sync only the results. Swallow single-letter accelerators while the Notes canvas is mid-recompose or a then= refocus is pending (event-driven, not timers, per ADR-104). |
| S-09 | P2 | Library-shell | F6 never enters the Export or Search/RAG canvas, and entry focus stays on the rail | NEW | Add Export and Search/RAG F6 targets (library-export-name/destination, library-rag-query-input/run). _resolve_focus_target falls back to the pane's first visible focusable descendant. Entry focus lands in the canvas at every width. Contract test per rail row. |
| S-11 | P2 | Library-shell | Escape means different things per destination; the advertised 'esc focus rail' does nothing when the rail is collapsed | TASK-32650, TASK-31571, TASK-31569 | One rule: Escape from a list only moves focus, to Search Library… when Nav is open, or to the Nav grip (chip 'esc focus Nav grip') when it is collapsed. Esc leaves Media select. Esc in a filter keeps the draft. Put the Escape ladder in F1. |
| S-13 | P2 | Library-shell | Arrival focus and Tab order do not follow the screen: palette arrival focuses '⌃1 Home' (13 nav stops), Notes entry focuses the Nav grip | TASK-32649, TASK-32301 | On a real arrival (not a modal pop), focus the destination's entry target from its list-arrived event, never the Nav grip. Make the Notes tree a single Tab stop with arrows inside. |
| S-15 | P2 | Library-shell | Footer and F1 disagree; F1 has a key-less '- : typing in field' row and no 'how it works'; Search/RAG advertises an inert 'o open evidence' | NEW | F1 reads the footer tuple that was actually registered. Drop key-less rows and expand 'after esc:' into a group. Gate 'o' through check_action. Use one predicate for Conversations 'c', and refresh the footer when Conversations arrives. Add a 'How Notes works' block. |
| S-16 | P2 | Cross-destination | At 80 columns the Media reader's mode and action rows are cut mid-word ('Highligh', 'M'), so Info and More are off-screen | NEW | Mark the active mode with '✓ ' instead of ' (selected)', and wrap the toolbar instead of clipping it. Add an ellipsis plus tooltip on single-line Library labels. 80x24 label census test. |
| S-18 | P2 | Cross-destination | A deleted note stays staged in Console as 'Ready'; the send goes ahead without it and says nothing | NEW | When authorization drops staged references, block or show a notice at send: 'Trip planning: Lisbon was deleted — not sent · Remove · Restore'. Clear the reference. Revalidate the strip when entering Console. |
| S-20 | P2 | Cross-destination | Console 'Search Library' modal: typing goes nowhere, 'Staged' is shown before results exist, a jargon send-block, results re-stage after Un-stage | NEW | Set AUTO_FOCUS='#console-rag-settings-query'. Show 'Searching Library… · the first search loads the model' with Cancel, and let Send wait. Use plain block copy. Add a generation fence so Un-stage or Cancel drops late results (consider splitting this part out as P1). |
| S-21 | P2 | Library-shell | The landing's only 'Import…' goes to Media import, with no pointer to importing notes | NEW | Add a line to the Import media canvas: 'Markdown or Obsidian notes? Notes ▸ Add from files… keeps them editable' (as an action). Offer 'Import as notes / as media' when the selection is mostly .md. Make the landing tooltip honest. |
| L-15 | P3 | Media | Reader Find needs Enter and keeps showing the previous query's count and marks | NEW | Add an Input.Changed handler: while the input differs from the submitted query, the status reads 'Enter to find' and Prev/Next are dimmed. |
| L-24 | P3 | Search-RAG | Search/RAG silently keeps the previous query's source toggles | NEW | Make the results count scope-aware: '2 results · Media only — Search all sources'. The rail box restores all sources. |
| L-25 | P3 | Media | An absolute file:// path takes the reader's first rows, and the reader is no wider than the list | NEW | Show the byline as '<basename> · local file' and put the path in Info. Give the Reader a ~72-cell prose floor before list_grows. Collapse Nav automatically, once, when a document loads. |
| L-29 | P3 | Media | Selecting an item in Media Trash keeps an unrelated live item in the Reader | task-31635 | Show a content excerpt on the selected Trash row. Change the identity line to 'Still showing <title> from Media · r restore to read'. |
| L-35 | P3 | Ingest | A fully successful import still offers 'Retry this batch' and the footer 'r retry' | NEW | Make the label depend on the outcome: 'Retry this batch' only when rows failed or were skipped, otherwise 'Import again with these settings…' and 'r import again'. Change the button, binding and chip together. |
| L-36 | P3 | Library-shell | Help and empty states never say what Library, Prompts or Skills are | NEW | Add a one-sentence 'What this is' per surface via WorkbenchHelpState.notes. Define a prompt in the Prompts empty state. Change 'invocable: user & agent' to 'Can be run by: you or the agent'. Show the trust banner only when user-added skills exist. |
| L-37 | P3 | Media | Restore and archive reset 'updated' to now and reorder Newest and the landing | task-32307 | Sort and age conversations by their last message time. Widen task-32307 to keep pre-trash recency on restore. Leave the last_modified/ADR-147 stamp untouched. |
| L-38 | P3 | Media | Clearing the Media filter also resets sort and type chosen while filtered | NEW | On clear, rebuild the scope from the applied scope with query='' (keeping sort and type). Use the snapshot only to restore the selection. |
| N-31 | P3 | Notes | Notes text fields never show the 'typing in field' footer state, and Esc in the filter discards the unsubmitted draft | NEW | Apply the shared 'typing in field / after esc:' footer grammar to the Notes tiers. Esc in the filter keeps the draft. |
| N-33 | P3 | Notes | Ctrl+S saves only in Skills; the Notes save action is dead code; Prompts and Folder files have no save key | NEW | Converge on ADR-031: remove the Skills ctrl+s (library_screen.py:1047) and the dead action_library_notes_save. Or amend ADR-031 for an editor-scoped Ctrl+S bound in all four editors at once (owner question Q2). |
| N-34 | P3 | Notes-import | Escape cancels a running import while ‹ Notes leaves it running; every exit drops staged sync-review choices | NEW | Esc during an import behaves like ‹ Notes, and the import continues. Keep staged review choices until the observation changes. Give the back cue an origin ('‹ Sync folders'). Clear the stale 'Choice staged' line. |
| S-12 | P3 | Cross-destination | Active mode is invisible on Edit/Preview/Info, the 'Library notes \| Folder files' strip and Read/Info; the active Skills tab looks disabled | NEW | Prefix the active option of every mode or segment strip with '✓ ' (pad the others). Rename Skills' '-active' (Textual's press-flash class) to 'is-active' and give it a real style. |
| S-14 | P3 | Cross-destination | Six focus-indicator dialects: a one-cell underline, tints of 1.2–1.6:1, Skills rows without the █ bar, an amber Reader border against blue elsewhere | NEW | Use two dialects: a thick left edge for rows (extend task-31983 to Skills and Collections) and a ┃label┃ outline for 1-row buttons. The Reader uses $ds-action-focus. Scroll containers that are not the sole target are not focusable. Generalise the TASK-32613 stop-enumeration test. |
| S-22 | P3 | Library-shell | A relaunch drops the working context (staged sources, open note, filter, expanded folders) | NEW | Amend ADR-033 for an ids-only resume pointer (route, note id, expanded folders) that feeds the landing Continue slot. Never persist staged sources or filter text. |
| S-23 | P3 | Library-shell | Unexplained chrome: an 'Agent_Lessons' folder with no gloss once notes exist, and literal '[ ]' task boxes in Preview | TASK-32126, TASK-32355 | Always gloss Agent_Lessons (not only in an empty library) and add a '(empty — …)' child. Preview maps task items to ☐/☑ (read-only, with a legend line). Grips stay as ADR-086 specifies. |
| S-24 | P3 | Cross-destination | Internal vocabulary and raw exceptions on screen ('rail', 'placement', 'owner review', 'lane', 'Persisted source', 'PermissionError'); the export failure is shown three times | TASK-32573, TASK-32451, TASK-32378 | Use one name, 'Navigation' (footer 'esc focus navigation'). Rename to 'Folders' and 'Remove from folder'. Change the alarm to '! Sync folder inactive · open Manage sync folders'. Add a shared OSError-to-sentence helper. Show one message, not three. Add a copy-lint denylist test. |
| S-25 | P3 | Cross-destination | Polish: a toast covers its own Save/Discard buttons, receipts never name where deleted notes went, 'back to hub', no Study help labels | NEW | Receipt: '✓ deleted · <title> · in Recently deleted' (keep it, per ADR-055). Change 'back to hub' to 'back to Library' and add Study help labels. Dock toasts away from action rows. Use 'Next: Write the body below the title.' when a title exists. |
| S-26 | P3 | Library-shell | User Guide keyboard and label claims disagree with the live app (8 of 12 checked) | TASK-32590, TASK-32607, TASK-32589 | Rewrite the 8 sentences after the behaviour fixes land. Land 32589's drift gate. Give each behavioural claim an executable twin test at 80x24 and 120x36. |
| S-27 | P3 | Library-shell | The Items grip label goes stale across routes ('Items' painted on Notes, or 'Notes' on Media) | NEW | Add LibraryAdaptiveReaderPaneGrip.set_pane_label(), which syncs and calls refresh(). Mounted test. |
| S-28 | P3 | Library-shell | Select-mode and hand-off keys differ between Media and Notes (s/Space vs button/Enter; e only in Notes; c only in Media) | TASK-31569 | One select grammar on Media and Notes: s enters the mode, Space toggles (priority binding), e exports, Esc leaves. Gate with check_action and keep the footer 1:1 with bindings. |
