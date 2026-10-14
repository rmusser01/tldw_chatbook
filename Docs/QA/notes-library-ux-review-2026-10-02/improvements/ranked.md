# Library + Notes improvements: merged and ranked (2026-10-03)

Target: origin/dev `2d34cbf80d`, worktree `.worktrees/notes-library-ux-review`.
Inputs: three lens files with 38 ideas: `learnability.md` (LRN-01 to LRN-12), `power.md` (PWR-01 to PWR-13, numbered as in that file) and `ia-loop.md` (IA-01 to IA-13).
Findings: `../master-findings.md` lists 103 findings. **None were refuted**, so every N-, L- and S- id below counts as verified.
Constraints applied: `PRODUCT.md` principles 1-8 and its anti-references, `DESIGN.md` (Interaction Quality Contract, Destination Header, Don'ts), the ADR table in `../maps/known-context.md` §4, and the standing owner rule: redesigned screens follow the Console/Settings visual language, and each redesigned screen or mode needs explicit owner approval before implementation. In the tables, **GATE** marks an item that triggers that rule.

## How this was scored

- **Merging.** The 38 ideas became 30 ranked items, and the merge map at the end traces every idea. Where an idea held an S-effort part that fixes a P0 or P1 on its own, that part was split out as a quick win. Its larger remainder stays in *next* or *strategic*.
- **Score** = impact (1-5) x confidence (0.4-0.95) / effort (S = 1, M = 2, L = 4).
  - Impact weighs the severity and breadth of the verified findings an item removes. Removing a P0, or the core ingest→Console loop for every user, scores 5.
  - Confidence is high (≥0.85) when I re-read the cited code anchor at `2d34cbf80d` and the fix is local. It is lower when the item depends on another item, on a product ruling, or on an untested mechanism.
- **Buckets.**
  - *quick-win*: S effort and clear value.
  - *next*: M effort with no ADR change, although it may need an ADR-031 refinement note, as tasks 32458 and 33622.10 did.
  - *strategic*: L effort, **or** it needs an ADR amendment or new ADR, **or** it needs an owner product ruling before work can start.
- **Order.** Items are ranked by score within each bucket. Ties go to the item that removes more P0/P1 findings.
- **Anchors re-checked live in code at `2d34cbf80d`:**
  - `LibraryScreen` has `flush_pending_work` (`library_screen.py:11109`) but neither `confirm_quit` nor `prepare_for_quit`, and the quit flow calls only those two (`Widgets/confirmation_dialog.py:205-240`). R-01 is therefore a small fix.
  - The raw-id recovery copy is at `Workspaces/eligibility.py:79`, and the "Local Default" branch is at `Workspaces/display_state.py:582-600`.
  - The sync-root literal is at `library_notes_sync_controller.py:819`.
  - `LIBRARY_SUBROUTE_COMMANDS` holds only Artifacts and Skills (`app_command_providers.py:327-338`).
  - The Notes Preview `Markdown(...)` at `library_notes_canvas.py:2909` keeps Textual's default `open_links`. The in-repo precedent for `open_links=False` is `library_artifacts_widgets.py:198`.
  - `extract_note_link_targets` is at `ChaChaNotes_DB.py:208`.
  - Console's `ctrl+k` switcher is at `chat_screen.py:1936`, and the single pending launch is set at `UI/Console_Modules/retrieval.py:641`.
  - `WorkbenchHelpState.notes` exists, but Library does not fill it.

## Fix first: verified P0/P1 defects that no proposal fixes

These are bug fixes, not improvements. Each already has a concrete fix in `master-findings.md`, and they should ship before or alongside the quick wins. Several items below assume they are done.

| ID | Sev | Why no proposal covers it |
|---|---|---|
| L-01 | P0 | The Media list "Export…" freezes the app. No lens touched it. The fix is a `call_later`/worker hand-off (see the master entry). |
| L-02 | P0 | Analysis always says "No analysis provider is configured" because `load_settings()` drops `[analysis_defaults]`. R-16 fixes only the message that points users to the wrong place. |
| N-02 | P0 | The sync wedge at `notes_sync_executor.py:5384` (a postcondition that compares unlike things). R-10 stops the UI from lying about it but does not unwedge it. |
| N-04 | P1 | The Notes tree cannot scroll at ≥120 cols. This is a CSS scroll-owner fix. R-11 is only a workaround. |
| N-05 | P1 | The note editor header overflows at 120-200 cols, so Save and Use in Console are unreachable. |
| N-10 | P1 | The inline delete confirmation renders off-screen. |
| L-11 | P1 | The built-in skill "Enabled" switch shows no state. |
| S-07 | P1 | Closing F1 or any modal drops focus. |
| S-10 | P1 | Keyboard input stops at 80x24 after a resize removes the focused widget. |

Also unaddressed (P2/P3): N-28, N-30, L-14, L-17, L-20, L-21, L-28, L-32, L-37, L-38, S-14, S-27. Partially addressed: S-16 (R-07 covers only the rail heading), L-31 (R-23 removes 2-3 rows of workspace text), L-33 (R-22 sidesteps the download for the sample only).

## Ranked list

| Rank | Item | Bucket | Score | I | C | E | Verified findings addressed | Gate / ADR |
|---|---|---|---|---|---|---|---|---|
| R-01 | Quit and modals never lose a Library draft | quick-win | 4.75 | 5 | 0.95 | S | N-01, N-06 | ADR-027 seam only |
| R-02 | Media "Use in Console" links on use, as notes and conversations already do | quick-win | 4.25 | 5 | 0.85 | S | S-01, S-24 | none (existing link-on-use policy) |
| R-03 | Name sync roots and stop optimistic sync copy | quick-win | 3.60 | 4 | 0.90 | S | N-18, N-19, N-35 | none |
| R-04 | Name the model and scope before a paid Ask; replay history in its own mode | quick-win | 3.00 | 4 | 0.75 | S | L-06, L-27, L-24, L-23 | ADR-003/079 respected |
| R-05 | F1 says how each Library surface works | quick-win | 2.70 | 3 | 0.90 | S | S-15, L-36, S-11, N-33 | ADR-031 (advertised == working) |
| R-06 | Reach Notes by name: palette subroutes, More ▾, landing | quick-win | 2.55 | 3 | 0.85 | S | S-21, S-13, N-13 | ADR-015 (task-423 exception) |
| R-07 | Graduate the rail gradually | quick-win | 2.10 | 3 | 0.70 | S | L-36, S-16 | ADR-076 clarification note |
| R-08 | One input-safety contract for every Library field and editor | next | 2.13 | 5 | 0.85 | M | N-07, N-25, N-33, N-12, N-31, N-34, L-05, L-07, L-08, L-30, S-08 | ADR-031 refinement note; ADR-027 |
| R-09 | Console staged context accumulates, revalidates, never sends stale | next | 1.88 | 5 | 0.75 | M | S-06, S-18, S-19, S-20, L-09 | **GATE** (Console Inspector); ADR-079 |
| R-10 | One sync-health projection read by every surface | next | 1.75 | 5 | 0.70 | M | N-02, N-03, N-15, N-16, N-17 | ADR-073/059 unchanged |
| R-11 | Ctrl+K "Go to…" for every Library item, with search-or-create | next | 1.60 | 4 | 0.80 | M | N-21, N-04, S-09, S-13, S-22, L-03 | **GATE** (new modal); ADR-031 refinement note |
| R-12 | Export that keeps provenance and never overwrites | next | 1.60 | 4 | 0.80 | M | N-11, L-04, N-20, L-18, N-36 | light GATE if canvas layout changes |
| R-13 | One query grammar for every Library filter, then saved views on it | next | 1.50 | 4 | 0.75 | M | N-09, L-13, L-15, L-34, S-22 | TASK-19558 measured change; **GATE** for saved-view rail rows |
| R-14 | Link-native notes | next | 1.50 | 4 | 0.75 | M | N-08, N-26, N-21, S-03 | migration; ADR-073/021 bytes |
| R-15 | Capture with provenance from anywhere | next | 1.40 | 4 | 0.70 | M | S-02, S-03, S-04, N-13, L-34, L-07, L-08 | **GATE** (capture modal); ADR-027 |
| R-16 | No dead ends: every blocked control names its prerequisite and offers it | next | 1.40 | 4 | 0.70 | M | N-15, N-17, N-24, N-16, L-22, S-17 | ADR-073 for sync exits |
| R-17 | Zero states that perform the first action | next | 1.40 | 4 | 0.70 | M | L-12, N-37, L-36, L-34, S-21 | **GATE** (each empty state) |
| R-18 | Carry the first loop past graduation | next | 1.40 | 4 | 0.70 | M | S-21, L-35, L-13, L-36 | ADR-076 note; light **GATE** (landing strip) |
| R-19 | One Library vocabulary, a Terms list, and a copy lint | next | 1.20 | 3 | 0.80 | M | S-24, S-23, N-35, S-20, S-25, L-36 | none |
| R-20 | Resume where you left off, with Back | next | 1.13 | 3 | 0.75 | M | S-22, L-16, N-27, S-02, N-21 | ADR-104, ADR-086 |
| R-21 | Study from chosen sources, with an honest capability line | next | 1.05 | 3 | 0.70 | M | S-05 | **GATE** (Study hand-off canvas) |
| R-22 | A removable sample to learn on | next | 0.50 | 2 | 0.50 | M | L-36, L-10 | ADR-076; owner should confirm shipping content |
| R-23 | The Default workspace means "everything local"; workspaces appear at two | strategic | 2.00 | 5 | 0.80 | M | S-01, S-20, S-24, L-31 | **ADR-027 amendment**; owner ruling |
| R-24 | One answering engine: Library questions become Console turns | strategic | 1.20 | 4 | 0.60 | M | L-06, L-27, S-04, L-23 | **owner product ruling**; **GATE** |
| R-25 | One door in, routed by outcome | strategic | 0.90 | 3 | 0.60 | M | S-21, N-17, N-37, L-34, L-36 | reopens the **task-32627 deferral**; **GATE** |
| R-26 | Context bar in the Library header | strategic | 0.90 | 3 | 0.60 | M | S-17, S-19, L-24, N-02, S-01 | depends on R-09/R-10/R-23; **GATE** |
| R-27 | One Library verb grammar, generated into footer, F1, palette and guide | strategic | 0.88 | 5 | 0.70 | L | S-28, S-11, S-12, S-13, S-15, S-26, S-08, N-22, N-31, N-32, L-03, L-09, L-26, L-29 | **owner ruling on key semantics**; ADR-031 refinement |
| R-28 | Reading desk on an indexed note→source relation | strategic | 0.81 | 5 | 0.65 | L | S-02, S-03, S-04, L-16, L-25, L-34 | **ADR-086/084 amendment**, **ADR-105 ruling**; **GATE** |
| R-29 | A selection that survives paging and filtering, with real bulk verbs | strategic | 0.75 | 4 | 0.75 | L | N-14, N-23, N-29, L-19, L-09, S-06, S-28 | ADR-055/067; **GATE** |
| R-30 | Templates are notes, plus "Today" | strategic | 0.40 | 2 | 0.40 | M | none verified | owner ruling on scope |

---

## Quick wins

### R-01 · Quit and modals never lose a Library draft (from LRN-02 rule 4, PWR-13 parts 2 and 5)

- **Change:**
  - Add `LibraryScreen.confirm_quit()` and `prepare_for_quit()`, both awaiting the existing `flush_pending_work()` (`library_screen.py:11109`). The quit protocol in `Widgets/confirmation_dialog.py:205-240` already calls these on every screen, and Library is the screen that lacks them.
  - When the flush is vetoed or fails, show "Quit and discard unsaved changes to "<title>"? Stay · Discard" with Stay focused. Show it through `await_quit_prompt`, so the prompt cannot hang the quit worker.
  - On screen resume after a modal pop, re-arm the note autosave, or flush on suspend. Derive "saves automatically" from whether a save is scheduled, not from dirty state.
- **Why first:** N-01 is a P0 data loss reproduced by four sources. N-06 is a standing false "saves automatically" promise. Both fixes live in one screen class and need two Pilot tests.
- **Constraints:**
  - ADR-027: the flush goes only through the note-session coordinator, with no second save path.
  - ADR-031 refinement 33622.10 covers the quit flow.
  - No visual redesign, so no owner gate.

### R-02 · Media "Use in Console" links on use, as notes and conversations already do (from LRN-01 part 3, IA-02 part 3)

- **Change:**
  - Pressing Media reader "Use in Console" or `c` links the item to the active workspace, stages it, and leaves "✓ Linked to Default · staged in Console · Undo link". This is the conversation receipt from task-32107 with the task-32388 Undo.
  - Search `u` follows the same rule.
  - `eligibility.py:79` stops interpolating `workspace-default` and uses the display name.
- **Why:** S-01 is the first-run dead end on PRODUCT.md's core loop. j2, j4 and j7 all hit it. Today registry links are written for notes (`library_notes_controller.py:4594-4616`) and conversations (`library_screen.py:13906`) but never for media. This item brings media into line with the house policy without any ADR change.
- **Constraints:**
  - Server-backed items must still refuse honestly, beside the action that fixes the problem (`_LINKABLE_REASON_LABELS`, task-32056).
  - R-23 later removes the need for the link while Default is the only workspace.

### R-03 · Name sync roots and stop optimistic sync copy (from PWR-11, IA-05)

- **Change:**
  - Title each root with its folder basename plus path, e.g. "Vault3 · ~/Notes/Vault3", replacing the literal at `library_notes_sync_controller.py:819` (task 32451, open). Prefix receipts with the root name.
  - When the history read fails, show "History unavailable — Retry", not "No writes yet." (N-19).
  - Replace raw nanosecond timestamps and exception class names with local times and plain reasons (N-35).
- **Why:** N-18 and N-19 are P1. With two roots, Pause, Resume and Recovery can target the wrong folder. The change is copy and data only, with no ADR change.

### R-04 · Name the model and scope before a paid Ask; replay history in its own mode (from LRN-08 parts 1, 3, 5, 6; IA-06 part 3; IA-07's pre-run line)

- **Change:**
  1. Fix the provider/model resolution in `library_rag_answer_service.py:163-189` so it uses the configured model. This is L-06, and it is a precondition.
  2. Before the run, the mode toggle shows the model that will be billed: "RAG Answer · sends question + top passages to OpenAI gpt-4.1-mini".
  3. Recent searches store their mode and scope, mark paid entries "RAG · paid", and replay exactly as saved.
  4. Searches started from the rail box reset scope to All sources and show the scope line ("2 results · Media only — All sources").
  5. On arrival, the answer and the evidence scroll into view (L-23).
- **Constraints:**
  - Never auto-run a paid call (PRODUCT principle 8).
  - ADR-003: per-run choices stay in Library. ADR-079 is untouched.
  - Keep the "RAG Answer" label until R-19/R-24 settle the rename in one pass across Library, Console and the guide.

### R-05 · F1 says how each Library surface works (from LRN-05)

- **Change:**
  - Give each surface a 3-6 line "How <surface> works" block, derived from copy the surface already renders, the way Settings' `_category_help_notes` does (`settings_screen.py:4227`, TASK-23110). It fills the existing `WorkbenchHelpState.notes`. End each block with "Esc here: …".
  - Fix the malformed "- : typing in field" row and drop the inert "o open evidence" chip.
  - Add a test that every `_LIBRARY_HELP_SURFACE_LABELS` surface has notes and that every listed key passes `check_action`.
- **Constraints:**
  - Help states today's truth, e.g. "a typed [[Title]] stays plain text" until R-14 lands.
  - Lazy import (ADR-097).
  - Follows the Settings help visual language, so no owner gate.

### R-06 · Reach Notes by name: palette subroutes, More ▾, landing (from LRN-10 parts 1, 3, 4)

- **Change:**
  - Extend `LIBRARY_SUBROUTE_COMMANDS` (`app_command_providers.py:327-338`) with Notes, Media, Conversations, Prompts, Collections, Search/RAG and Import. The palette query "notes" must rank "Library — Notes" first.
  - Each command lands with focus on the first list row (S-13).
  - Add "Notes (in Library)" to More ▾ and "Notes (N)" to the landing quick actions.
- **Why:** j1's first two actions were to look for Notes in the nav bar and in More ▾.
- **Constraints:**
  - ADR-015: subroutes are task-423's labelled exception, so this adds no destinations.
  - The arrival must not reproduce N-13's blank tree. Fix N-13's defect alongside this.
  - The "Go to note…" switcher from LRN-10 part 2 moves to R-11. That keeps `o` meaning "open evidence".

### R-07 · Graduate the rail gradually (from LRN-12, plus LRN-06 part 5)

- **Change:**
  - On the first graduated render, collapse the sections whose rows are all zero, with a summary such as "Study · nothing yet ▸".
  - A section expands once, the first time it gets content. After that, `library.rail_state.sections` (the user's choice) always wins.
  - Section glosses move from width-dropped row suffixes to a second line under each heading.
  - The heading reads "Navigation" at every supported width.
- **Constraints:**
  - ADR-076: this changes default section state only. Graduation stays permanent and silent. Record a clarification note.
  - ADR-034 owns the ▸/▾ glyphs.
  - Collapsed sections keep their counts and stay keyboard-reachable.

## Next

### R-08 · One input-safety contract for every Library field and editor (from LRN-02 rules 1-3, 5-6; PWR-13 parts 1, 3, 4, 6)

- **Change:** add six rules to DESIGN.md's Interaction Quality Contract and enforce them with one parametrized Pilot suite, one test per field kind:
  1. Normalize instead of vetoing (trim the title, dedupe keywords, expand `~`), and say so inline: "Removed a trailing space from the title". This covers N-07, N-25 and L-05.
  2. Autosave validation never moves focus.
  3. Escape never destroys typed text. The first Esc on a dirty draft shows "Unsaved — Save · Discard" (ADR-055 pattern D). This covers L-08 and N-34.
  4. Cancel is a neutral outcome, and focus returns to the opener (N-12).
  5. Single letters act only when focus is on a row, reader or chrome, never while focus is None or within one re-render of an Input submit. This covers S-08, L-07 and N-31.
  6. Prompts and Skills answer Escape with "Save · Discard · Keep editing". Retire the Skills `ctrl+s` (`library_screen.py:1047`) to conform with ADR-031 rule 2 (N-33).
- **Optional phase 2:** a draft journal inside the coordinator, plus "Recovered unsaved text from 20:11 — Keep · Discard".
- **Constraints:**
  - ADR-027: one coordinator, with the journal inside it.
  - ADR-029: a recovered Folder files draft never overwrites a file that changed on disk.
  - The draft store is user data, so it needs backup coverage and pruning.
  - Rule 5 is an ADR-031 refinement note, not a new rule. Rule 3 also covers Esc on a running import (L-30): back, not cancel.
  - No visual redesign.

### R-09 · Console staged context accumulates, revalidates, and never sends stale (from IA-01's accumulating list, PWR-05's "c appends", S-06's fix)

- **Change:**
  - Replace Console's single pending launch (`_set_pending_launch`, `retrieval.py:641`) with a Console-owned list of `EvidenceReference`s. `EvidenceBundle` already holds many (`build_library_rag_evidence_bundle`, `retrieval.py:685`).
  - Each hand-off appends, de-duplicated by canonical id, with the toast "Added 'paper-retrieval-practice' · Console context: 3".
  - Rows are revalidated at send: "⚠ note deleted — will not be sent · Remove" (S-18).
  - The strip, Inspector and status bar read the same list (S-19).
  - "Staged" is never shown before results exist (S-20).
  - A verb never acts on an item the filter has hidden (L-09).
- **Constraints:**
  - ADR-079: a manual mechanism only. Never turn on auto-retrieve or assistant Library tools.
  - Hand-off channels are typed single slots (`UI/Navigation/pending_handoff_store.py`), so the list must be a Console-owned store.
  - Show a visible size and a cap.
  - **GATE:** this changes the Console Inspector's Sources section. Follow the Console Inspector's existing visual language and get owner approval.
  - The verb renaming in IA-01 ("Add to context" / `a` "Ask in Console" / retiring the rail's whole-Library hand-off) is deliberately left to R-27, because it needs a key-semantics ruling.

### R-10 · One sync-health projection read by every surface (from IA-05, PWR-11's honesty and exit parts, LRN-09 parts 1-2, LRN-07's sync exits)

- **Change:**
  - One projection per root (name, path, state, last check, one working next action), rendered in the same words on:
    - the tree row: "▸ Vault3 ⚠ not syncing — review"
    - the editor's "where this note lives" line (task-32640): "In synced folder Vault3 · ⚠ not syncing since 20:33 · Review"
    - the Manage sync folders heading
  - Invariants, each with a test:
    - No surface renders a healthier state than the projection.
    - "In sync" requires a check newer than the last local change.
    - One blocked file blocks only itself, with a plain reason (N-16).
    - Every non-green state offers "Pause and keep both copies".
    - Review rows for deleted or renamed files get a resolution (N-15).
    - Add from files always opens at step one (N-17).
- **Constraints:**
  - It does **not** fix the N-02 wedge, which is listed under Fix first.
  - "In sync" can only be truthful once task-32633's 17 write paths signal.
  - ADR-073: no automatic winner. Pause keeps both copies and resolves nothing.
  - ADR-059 is unchanged. Keeping the vault's folder structure (PWR-11's last part) is deferred, because it needs an ADR-059 amendment (`notes.md:1205-1209`).
  - Updates are event-driven, not per-row polling (tasks 281, 33286).
  - Folder files needs parallel states (ADR-021/029).

### R-11 · Ctrl+K "Go to…" for every Library item, with search-or-create (from PWR-01, IA-06 part 2, LRN-10 part 2)

- **Change:**
  - A modal with the same key and grammar as Console's session switcher (`chat_screen.py:1936`; `console_prompt_picker_modal.py` is the smaller template).
  - The empty query shows Recent and Places. Typing matches titles, aliases and keywords, prefix-tolerant, with at most 8 hits per type.
  - Enter opens the item **and moves the list highlight onto it** (L-03).
  - The last row is 'New note "<query>"', created in Inbox.
- **Why:** opening a known note costs about 11 keys today (N-21). Reaching another canvas costs 6-29 Tabs (S-09). It also routes around N-04 until that defect is fixed.
- **Constraints:**
  - Record an ADR-031 refinement note: Ctrl+K means Go to/Switch in every destination.
  - ADR-067: bounded, source-owned top-N.
  - ADR-097: lazy import.
  - ADR-104: event-driven focus return.
  - Duplicate titles show their folder.
  - Body full-text search stays in Search/RAG.
  - **GATE:** a new modal, but it reuses Console's switcher visual language by construction.

### R-12 · Export that keeps provenance and never overwrites (from PWR-10, IA-12)

- **Phase 1** (defect fixes, each S; they could ship as quick wins):
  - Write front matter with a real YAML emitter (N-20).
  - Keep keywords in bundle manifests (L-04).
  - Before writing, ask "Add -2 / Replace" when a name collides, for note, prompt and same-day bundles (N-11, L-18).
  - Replace the canvas's "copies full media files" with an honest fidelity line: "Includes: text ✓ · keywords ✓ · original files ✗ (text only)".
- **Phase 2:**
  - Add "Export as Markdown folder": one .md per note, deterministic slugs, `[[Title]]` links, a `## References` section for sources, and a "4 changed · 2 new · Write" plan before re-export.
  - Session Git gains "View diff" for the selected row and for the staged set (N-36), reusing the conflict-compare box.
- **Constraints:**
  - Round-trip tests from creator to importer (`chatbook_creator.py:1454/1620`, `chatbook_importer.py:2772`).
  - Absolute paths are off by default.
  - Say "a one-time copy · later edits are not synced".
  - ADR-038/039: no automatic push.
  - Light **GATE** if the Export canvas layout changes.

### R-13 · One query grammar for every Library filter, then saved views on it (from PWR-04, IA-06 parts 1 and 4, LRN-08 part 2, PWR-08)

- **Change:**
  - One `LibraryQuery` parser for every list filter and for Ctrl+K: words ANDed in any order, the last word a prefix, plus `"phrase"`, `-word`, `#keyword`, `in:Folder` and `type:`. Filtering is live, debounced the way Media's is today.
  - Each result shows what matched.
  - Zero results offer a relaxation: 'No notes match "Lisb zebra" · 3 match "Lisb"'.
  - A plain question that matches nothing offers "[Ask it] (sends to <model>)" and never auto-runs it (L-13).
  - Reader Find updates live and drops the stale count (L-15).
  - Phase 2: saved views as rail rows under each destination, including a built-in "Read later (N)" that closes L-34's three-way split, and a remembered filter (S-22).
- **Constraints:**
  - Notes is phrase-only by the deliberate TASK-19558 rule (`fts5_match_forms.py:368-385`). Ship this as the measured change that rule asks for, with a before/after result-count table.
  - The builders exist (`:242` prefix, `:348` AND).
  - `notes_fts` covers only title and content, so keywords come in through a join.
  - ADR-013: safe quoting.
  - Measure prefix cost on a 10k-note profile before any FTS prefix index (that would need a migration and a plan pin captured with `sqlite_stat1` absent).
  - Per-keystroke cost (task 32804.11).
  - **GATE** for the saved-view rail rows. ADR-076: they appear only once saved, or when a built-in view is non-empty.

### R-14 · Link-native notes (from PWR-03, IA-03 part 4)

- **Phase 1 (S):**
  - Notes Preview uses `open_links=False` plus a `LinkClicked` handler (`library_notes_canvas.py:2909`; precedent `library_artifacts_widgets.py:198`). `note://` and `media://` then open in-app instead of in the OS URL handler.
  - `extract_note_link_targets` (`ChaChaNotes_DB.py:208`) resolves a bare `[[Title]]` by unique, case-insensitive title.
- **Phase 2:**
  - `[[` completion, with "Create "X"" as the last row.
  - Info shows "Links out (N) · Linked from (N) · Unresolved (N)", fixing N-26's stale backlinks.
  - Following a link is reversible through R-20's Back.
- **Constraints:**
  - Storing unresolved targets needs a migration, a `VALID_TABLES` entry and a plan pin.
  - ADR-073/021: never rewrite bytes, and plain `[[Title]]` is the authored default, so synced Obsidian files stay clean.
  - Completion goes through the ADR-027 coordinator.
  - Ambiguous titles stay unresolved and are never guessed.
  - The rename-update prompt is opt-in.

### R-15 · Capture with provenance from anywhere (from PWR-06, IA-03 parts 2 and 5, IA-07 part 4)

- **Change:**
  - One capture modal, about 8 rows, that never navigates away. It opens from the palette, from `n` in readers (prefilled "About: [title](media://26)"), from `q` on a reader selection (block quote plus source link), and from Console "Capture as note", which then keeps the question, model, date and "Sources:" (S-04).
  - Escape keeps the capture. A receipt reads 'Saved "Lisbon packing" to Inbox · Open'.
  - Hand-written "Add analysis" becomes "New note about this" on the coordinator-backed editor. That retires the bespoke TextArea behind L-07 and L-08.
- **Constraints:**
  - The `media://` form exists today only in `MCP/resources.py`. In-app resolution comes from R-14 phase 1, so R-14 goes first.
  - Reuse `_capture_console_answer_as_note` (`UI/Console_Modules/message.py:2169`), with no second note writer (ADR-027).
  - ADR-104: focus return. ADR-031: no Ctrl+S.
  - **GATE:** a new modal, styled on Console's modal grammar.
  - Key conflict: PWR-07 and IA-04 want `n` to open the companion note. Use `n` = capture until R-28 is approved.

### R-16 · No dead ends: every blocked control names its prerequisite and offers it (from LRN-07, IA-02's blocked-control rule)

- **Change:**
  - Make the TASK-716 pressable-blocked pattern (`library_entry_canvases.py:416-421`) universal: "<Action> needs <prerequisite> — [fix]".
  - Identical reasons collapse into one line per group.
  - Hide what can never work. Concrete cases: N-15's dim "Apply reviewed"; N-24's "refresh and try aga…" with no refresh control; L-22's inert "Open original"/"Open manager"; S-17's silent vetoed nav switch, which gets a toast and re-selects the current destination; N-17 and N-16's recovery that never loops back to a failing button.
  - Add an architecture test asserting that every disabled Library button has a visible, non-tooltip reason (DESIGN.md: "Don't hide why an action is disabled").
- **Constraints:**
  - L-02's root bug is under Fix first. This item fixes only its pointer to a Settings page that cannot help.
  - Focusable disabled controls add Tab stops, so focus the grouped reason line instead.
  - Sync exits follow ADR-073 (pause, never resolve).

### R-17 · Zero states that perform the first action (from LRN-04)

- **Change:**
  - Each source owns one zero-state grammar: a definition in user words, one focused primary action, at most two secondary actions, and what becomes possible next. For example, Notes (0) shows "New note (n) · Bring in Markdown files… · Edit a folder in place…", and the work pane explains autosave, Preview and Use in Console.
  - Controls that act on nothing (Filter, Sort, Select, Export) render from the first item onward.
  - Media, Conversations and Skills open list-first at 64-98 cols (L-12, P1).
  - Folder files unlinked gets a centred "Edit Markdown files where they already are — nothing is copied · Choose folder…" (N-37).
  - The Collections box is labelled "Note (optional)" (L-34).
- **Constraints:**
  - Copy lives in each source's state module, because ADR-076 rejected a shared empty-state compositor.
  - Notes list action budget (`library_notes_canvas.py:125-139`).
  - List-first must keep ADR-086's permanent work pane.
  - Overlaps open task 32304 (<64 cols).
  - **GATE:** each redesigned empty state is a mode change needing approval.

### R-18 · Carry the first loop past graduation (from LRN-03)

- **Change:**
  - After an all-success import, the queue shows 'Next: Find it — search what you just imported · Ask about "<title>"'. Retry appears only when a row failed (L-35).
  - "Find it" opens Search with the scope chip "Just imported (N)".
  - The first evidence card shows "Next: Use in Console (u)".
  - The expanded landing carries a one-row "First loop 1/3" strip until the loop is done or hidden. It completes from real events.
- **Why:** today one import silently replaces Get started, so steps 2-3 are never seen (S-21, L-36). Plain questions meet "No evidence matched" (L-13).
- **Constraints:**
  - ADR-076: no wizard and no mode. Graduation stays permanent and silent. The persisted "first loop done/hidden" key needs an ADR-076 note.
  - Until R-02 lands, step 3 must route through Search evidence.
  - Light **GATE** for the landing strip.

### R-19 · One Library vocabulary, a Terms list, and a copy lint (from LRN-06)

- **Change:**
  - A `LIBRARY_TERMS` table of about 22 rows, each holding a term, a one-sentence definition, where to manage it, and banned synonyms.
  - F1 adds "Terms on this screen", built on R-05.
  - The palette gets "Library term: <term>" commands.
  - A copy-lint test, seeded from B's `s2_vocab.py`, bans internal words (rail, placement, lane, cutover, authority, exception class names) in user-facing strings under `Widgets/Library`, `UI/Library_Modules` and `Library/*_state.py`. It also enforces one name for the left pane. This covers S-24, S-23 (Agent_Lessons gloss), N-35, S-20's "Review source authority", and S-25's "hub".
- **Constraints:**
  - The lint is the primary control, so the glossary stays short.
  - Load lazily (ADR-097).
  - Any "RAG Answer" → "Ask" rename happens in one pass with R-04 and R-24.

### R-20 · Resume where you left off, with Back (from PWR-12, IA-10)

- **Change:**
  - Each rail row keeps its own content context: open item, tab, scroll, caret, filter, expanded folders. Switching rows then closes nothing, so "loaded" is true (L-16, S-02).
  - The last item, filter and expanded branches persist per profile across relaunch (S-22).
  - A Back/Forward history covers items opened in Library, including followed links (N-21).
  - Open-by-id expands and reveals its target (N-27).
- **Constraints:**
  - ADR-104: event-driven, two-gate, authority-fenced restore with no timers.
  - ADR-086: geometry stays transient, and only content context persists.
  - Deep links beat remembered state (ADR-076).
  - Per-profile storage with backup coverage.
  - Key conflict: PWR-12 implies Backspace and IA-10 proposes Alt+←/→, which is unreliable across terminals. Ship palette "Back"/"Forward" now and settle the key in R-27.

### R-21 · Study from chosen sources, with an honest capability line (from IA-11)

- **Change:**
  - "Make study set…" from the note editor, any reader or any selection carries exactly the chosen items. `StudyScopeContext.source_items` already supports this (`study_scope_models.py:83-94`).
  - Before the button, one capability line: in local mode "Card generation needs a server here — write cards by hand with the sources beside you", with "Write cards…" as primary.
  - The whole-Library snapshot becomes an explicit "Use all of Library".
  - Esc returns to the originating item.
- **Constraints:**
  - Local generation is excluded. It is a new capability needing principle-8 cost disclosure, and it would make this L.
  - Keep Study a destination (anti-reference "study-only app").
  - **GATE** for the Study hand-off canvas.

### R-22 · A removable sample to learn on (from LRN-11)

- **Change:**
  - "Try with a sample" imports an offline pack labelled "Sample · remove anytime": two documents, two linked notes and one prompt.
  - Find works offline with no embedding download.
  - "Remove sample" is one ADR-055 receipt with Undo.
- **Why it ranks low:** no verified finding shows that the lack of a sample blocks anyone. It offers a safe alternative to L-10's demo, but L-10 still needs its own disclosure before creating anything.
- **Constraints:**
  - ADR-076 already treats sample content as non-graduating.
  - No network calls, and excluded from Continue once real content exists.
  - Needs a drift test.
  - The owner should confirm shipping bundled content.

## Strategic

### R-23 · The Default workspace means "everything local"; workspaces appear at two (from LRN-01, IA-02, IA-08's workspace chip)

- **Change:**
  - While Default is the only workspace, every local item is eligible: eligibility routes through the "Local Default" branch (`Workspaces/display_state.py:582-600`). Library then never says "workspace". The Details Workspace row and the Conversations workspace paragraph disappear, which also removes 2-3 rows of L-31's chrome. S-20's "Review source authority" send-block is not reached for local items.
  - With two or more workspaces, add a header switcher that shares Console's control and state, row markers "· in Thesis", and blocked controls that read "Use in Console · not in Thesis — [Link to Thesis and use]" with Undo.
  - A one-sentence explanation goes in the New Workspace dialog.
- **Why strategic:** the highest-value structural change. It extends ADR-027 (default-workspace chats) to Library sources, which needs an **ADR-027 amendment** (or a short new ADR). It also needs an **owner ruling** that the eligibility gate is scoping, not a security boundary. R-02 delivers most of the first-run value now, without that ruling.
- **Constraints:**
  - ADR-028: Default stays tool-less.
  - Server-backed items still refuse honestly.
  - Moving from one workspace to two must keep Default's items usable.
  - PRODUCT principle 7.

### R-24 · One answering engine: Library questions become Console turns (from IA-07, LRN-08 part 4)

- **Change:**
  - Library Search/RAG becomes free retrieval with multi-select evidence plus one paid button, "Ask in Console · OpenAI gpt-4.1-mini · question + 3 passages · ≈$0.001". It opens a Console conversation with the passages staged through R-09.
  - Readers get "Ask about this", scoped by the existing `EffectiveScope` id allowlist (`library_local_rag_search_service.py:433-453`).
  - Recent questions open their conversation and never re-run.
  - Option B keeps an inline answer but persists it through Console's pipeline.
- **Why strategic:** retiring the inline Library answer is a product call (**owner ruling**). PRODUCT principle 1 ("Console is the live work surface") supports it. R-04 removes the money-and-honesty harm in the meantime.
- **Constraints:**
  - ADR-079: never enables auto-retrieve or tools.
  - Console must carry Library's "answer does not cite staged evidence" caution.
  - **GATE** for the Search/RAG mode change.

### R-25 · One door in, routed by outcome (from IA-13, LRN-09 parts 3-4)

- **Change:**
  - One "Add to Library…" chooser that names outcomes, each with a one-line consequence:
    - Read and cite → Media
    - Edit as notes → Import once
    - Keep in sync → Keep synced
    - Edit where they are → Folder files
    - Save a web page → Collections
  - It recommends a route from what was pasted ("12 Markdown files — Edit as notes is recommended").
  - It always opens at step one, so N-17's stale phase cannot recur.
- **Why strategic:** folding the three notes worlds into one screen was explicitly **deferred by task-32627**, so it needs an owner ruling to reopen. A **phase 0 needs no ruling** and respects the deferral:
  - Add LRN-09's three-row consequence table ("Where the text lives / Edits made outside Chatbook") to the existing Add from files chooser.
  - Add "Bringing in Markdown notes? [Add to Notes…]" to Import media and the empty landing. This closes S-21.
- **Constraints:**
  - ADR-076: one chooser, never a wizard.
  - ADR-113/059: authorities stay separate.
  - Keep Import media's pre-check summary, a strength in j2 and j7.
  - **GATE.**

### R-26 · Context bar in the Library header (from IA-08)

- **Change:**
  - The Library header reads "Library | Local · Workspace: Thesis ▾ · Console context: 3 ▸ · Notes sync ⚠ 1 ▸". Each fact is a focusable button, and each is silent at its default.
  - A vetoed navigation shows its reason here and re-selects the current destination (S-17). That part can ship alone as a defect fix.
- **Why strategic:** each fact reads a projection that R-09, R-10 or R-23 creates. Built before them, it would be a third, disagreeing copy of the state. It also changes the ADR-015 header contract, which fits DESIGN.md's Destination Header role (readiness and authority).
- **Constraints:**
  - Anti-reference "control-room theater": at most four facts, silent by default, text never colour-only.
  - Verify the fit live at 80x24 and 120x36.
  - ADR-097: lazy.
  - **GATE.**

### R-27 · One Library verb grammar, generated into footer, F1, palette and guide (from PWR-02, IA-01's five verbs, IA-09's key grammar)

- **Change:**
  - A declarative `LibraryVerb` registry generates `check_action`, footer chips, F1, palette entries and the User Guide keyboard section. The guide section becomes a derived artifact checked by `preflight.sh`, which fixes S-26 by construction.
  - The same keys on every list: `/`, ↑↓ including folder rows (N-22), ←/→ collapse and expand, `]`/`[`, `1/2/3` for modes shown as "✓ Edit" (S-12), `s`/Space selection, `e`, `f`, `t`, `y`.
  - Three rules travel with the table:
    - **Target:** a verb acts on the selection, else on the highlighted row. It is disabled when the open item is filtered out (L-03, L-09, L-29).
    - **Focus:** no letter dispatch while focus is None (S-08, N-31).
    - **Escape:** field → list → rail, never destructive (S-11).
  - F6 leaves the note body (N-32).
  - Arrival focuses the first row (S-13).
  - Prompt insert keeps Instructions (L-26).
- **Why strategic:** L effort across `library_screen.py` (bindings at `:1010-1222`, gates at `:25253-25681`, footer sets at `:1243-1600`). It also needs an **owner ruling on key semantics** where the lenses conflict:
  - IA-01 makes `c` "add and stay" and `a` "Ask in Console". PWR-02 makes `c` "send, appending", and PWR-05 makes `a` "select all matching".
  - `c` stops meaning Resume on Conversations.
  - Backspace vs Alt+← for Back.
  - u/o aliases for one release.
  - Record it as an ADR-031 refinement: no Ctrl+S, and the destructive `t`/`x` keep their confirmation.
  - Do it after R-08, which removes the focus-race class first.

### R-28 · Reading desk on an indexed note→source relation (from PWR-07, IA-04, IA-03 parts 1 and 3)

- **Change:**
  - In Media, Conversations and Collections readers, `n` opens the item's companion note beside the reader. The reader keeps its scroll and width priority. Below the threshold the two stack, with a one-row header naming both (S-02, L-16).
  - `q` quotes a selection with its source link into the note (S-03).
  - Notes-to-source edges are indexed, so item Info shows "Notes about this (N)" and note Info shows "Sources (N)". This replaces Collections' raw "Note ID" box (L-34), and kept answers trace back (S-04).
  - Shrinking the file:// path block helps L-25.
- **Why strategic:**
  - An **ADR-086 amendment**: the permanent single-slot work pane gains a destination-owned companion slot with its own collapse order. Also an ADR-084 check that the Reader stays the width priority.
  - The edges need a migration, a `VALID_TABLES` entry and a plan pin, plus an **ADR-105 ruling** (portable organization or device-local).
  - It is a new layout, so **GATE**.
- **Constraints:**
  - The lenses disagree on the threshold (≥140 work-pane cells vs ~100), so settle it in the amendment.
  - Server Collections notes stay server-authoritative (ADR-113).
  - Trashed targets offer Restore (ADR-055).
  - Lazy mount (ADR-097). Restore follows ADR-104.
  - Bare letters are inert while the companion has focus.

### R-29 · A selection that survives paging and filtering, with real bulk verbs (from PWR-05, IA-09)

- **Change:**
  - Each destination owns an id-set selection that survives paging, filters and rail switches. A chip reads "12 selected · 3 shown".
  - "Select all 37 matching" is materialised from the query.
  - Bulk verbs: File into… (a picker listing every folder, N-23), Keywords…, Use in Console (N), Export, Trash through the single ADR-055 seam (task-32635), and Delete forever in Trash (N-29).
  - Confirmations name hidden rows: "9 aren't shown by the current filter."
- **Why strategic:** L effort, spanning all list destinations and select modes (**GATE**). A **Notes-only M phase** can ship first without the cross-destination grammar: fix the "Select all N shown" miscount, Space toggles a row, task-32635's bulk Trash, and a picker listing every folder (N-14, N-23).
- **Constraints:**
  - ADR-055: one seam and one receipt.
  - ADR-067: per-source sets and explicit "matching vs loaded" counts.
  - Stale ids are dropped with a count.

### R-30 · Templates are notes, plus "Today" (from PWR-09)

- **Change:**
  - A Templates system folder whose notes are templates, using `{{date}}`-style variables with no code execution.
  - A Today button and a daily template.
  - Import recognises a vault's templates folder.
- **Why last:** no verified finding backs it. It rests on j3's task 6 and idea 5 and on the seed's 80 hand-made daily logs.
- **Constraints:**
  - Needs an owner scope call.
  - Variables must not collide with Prompts' `{{notes}}` (L-26).
  - Templates are excluded from search by default.
  - Creation goes through the ADR-027 coordinator.

---

## Dependencies and sequencing

1. Ship the **Fix first** defects (L-01, L-02, N-02, N-04, N-05, N-10, L-11, S-07, S-10) and R-01 to R-07.
2. Ship **R-08** before any new editor or modal (R-11, R-15). It removes the focus-race and Escape-destroys-text classes that a new surface would inherit.
3. Ship **R-14 phase 1** (in-app link routing) before **R-15** (`media://` provenance links) and **R-28**.
4. Ship **R-13** before saved views (its phase 2) and before R-11 shares its parser.
5. **R-10** is truthful only after task-32633 (write paths signal sync). Ship it with the N-02 fix.
6. Build the projections in **R-09**, **R-10** and **R-23** before **R-26**, which only displays them. Make the key ruling in **R-27** before IA-01's verb renames.

## Owner decisions this list asks for

- **ADR amendments:**
  - ADR-027 (R-23)
  - ADR-086/084 plus an ADR-105 ruling (R-28)
  - ADR-059 for structure-preserving sync (deferred out of R-10)
- **Product rulings:**
  - Retire or keep the inline Library RAG answer (R-24).
  - Reopen task-32627's three-worlds deferral (R-25).
  - Key semantics for `c`, `a` and Back (R-27).
  - Ship sample content (R-22).
  - Daily-notes scope (R-30).
- **Redesign approvals (GATE):** R-09, R-11, R-13 (saved views), R-15, R-17, R-18 (light), R-21, R-24, R-25, R-26, R-28, R-29, and R-12 if its canvas layout changes.
  - Each must follow the Console/Settings visual language. R-11 and R-15 reuse Console's modal grammar, and R-05 reuses Settings' help block.
- **ADR-031 refinement notes (routine):** R-08, R-11, R-27.

## Merge map (every source idea)

| Source idea | Lands in |
|---|---|
| LRN-01 Workspaces appear only at two | R-02 (link-on-use part), R-23 |
| LRN-02 Input-safety contract | R-01 (rule 4), R-08 |
| LRN-03 Carry the first loop | R-18 |
| LRN-04 Zero states | R-17 |
| LRN-05 F1 how it works | R-05 |
| LRN-06 Vocabulary and Terms | R-19 (rail glosses → R-07) |
| LRN-07 No dead ends | R-16 (sync exits → R-10) |
| LRN-08 Search vs Ask by consequence | R-04 (parts 1, 3, 5, 6), R-13 (part 2), R-24 (part 4) |
| LRN-09 Three notes worlds | R-10 (parts 1-2), R-25 (parts 3-4 as phase 0) |
| LRN-10 Find Notes | R-06 (parts 1, 3, 4), R-11 (part 2) |
| LRN-11 Sample | R-22 |
| LRN-12 Graduate the rail | R-07 |
| PWR-01 Ctrl+K Go to | R-11 |
| PWR-02 Verb grammar | R-27 |
| PWR-03 Link-native notes | R-14 |
| PWR-04 Query language | R-13 |
| PWR-05 Persistent selection | R-29 ("c appends" → R-09) |
| PWR-06 Capture sheet | R-15 |
| PWR-07 Reading desk | R-28 |
| PWR-08 Saved views | R-13 (phase 2) |
| PWR-09 Templates and Today | R-30 |
| PWR-10 Markdown export and Session Git diff | R-12 |
| PWR-11 Vault-grade sync | R-03 (names, history, copy), R-10 (health, per-file, exit); structure-preserving sync deferred (ADR-059) |
| PWR-12 Resume | R-20 |
| PWR-13 Ctrl+Q-proof editors | R-01, R-08 |
| IA-01 Hand-off vocabulary and accumulating context | R-09 (accumulating list), R-27 (verbs) |
| IA-02 Default workspace | R-02 (interim link policy), R-23 |
| IA-03 Source links | R-14 (part 4), R-15 (parts 2, 5), R-28 (parts 1, 3) |
| IA-04 Reading desk | R-28 |
| IA-05 Health projection | R-03, R-10 |
| IA-06 One Library search | R-04 (part 3), R-11 (part 2), R-13 (parts 1, 4) |
| IA-07 One answering engine | R-24 (pre-run line → R-04; capture provenance → R-15) |
| IA-08 Context bar | R-26 (workspace chip → R-23) |
| IA-09 One selection | R-29 (key grammar → R-27) |
| IA-10 Pick up where you left off | R-20 |
| IA-11 Study from chosen sources | R-21 |
| IA-12 Export with provenance | R-12 |
| IA-13 One door in | R-25 |
