# Library Decomposition Wave 8 — Notes Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the notes subsystem from `LibraryScreen` — the FINAL extraction wave. Rough 2026-09-08 measure: ~262 note-named methods / ~138 distinct flat `_library_note[s]_*` attribute names (media, the prior record, was 251/122). After this wave the screen is a shell + shell-owned surfaces, and phase C (the resident-canvas click-freeze fix, the program's founding motivation) begins as its own motivated series.

**Architecture:** Identical mechanics to the seven merged series. `backlog/docs/library-decomposition-recipe.md` (all §, as updated through wave 7: FIVE census spellings, three-member prune whitelist, ancestors-membership shape + generalized bare-self census, interpreter-parity and interleaved-round-robin disposition rules, unconditional isolated-worktree baselines) is the how. Templates: `library_media_state.py`/`library_media_controller.py` (newest, largest) for mechanics.

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (on dev, as corrected).

## Global Constraints

- Everything prior waves' Global Constraints said, verbatim, plus all recipe lessons through wave 7. Non-negotiables with wave-8-specific teeth:
  - **Wave-7 inheritance #1 — `on_<message>` handlers:** notes likely OWNS name-dispatched handlers (media had zero, so §4's third whitelist member is untested since prompts). The census must enumerate them explicitly and the prune whitelist must be exercised with evidence.
  - **Wave-7 inheritance #2 — the `canvas_sync.py` dotted branch:** notes is the third subsystem needing the runtime-f-string dotted branch (`f"_library_{kind}_row_selection"` at `canvas_sync.py:219` @ 5dd2e71cf's convention). Add the notes branch in the move commit AND its mutation-verified guard beside the conversations/media precedents in `Tests/UI/test_library_selection_updates.py` — the fifth-spelling census is a required pre-move step.
  - **Wave-7 inheritance #3 — shared-seam fixture hazard:** other subsystems' test fixtures may exercise shared shell seams that write notes fields (the Skills-test incident); the bypass census must include seams, not just direct field references.
  - **Born-lazy controller import** + preimport-closure suffix in the move commit; state module-level.
  - The one-directional binding-pin weakness (TASK-32013) is NOT fixed in this wave's move commits — but the move task must run the ad-hoc AST resolver both directions (listed→resolves AND referenced→listed) as its own check, since that is what caught all nine missing bindings across wave 7's reconciliations.
- Known notes-specific facts to respect:
  - FOUR prior-extracted wiring modules: `library_note_import_controller.py`, `library_notes_sync_controller.py`, `library_notes_work_session.py`, `note_session_port.py` — untouched; their screen instances and delegating methods are exclusion candidates.
  - Notes owns the bidirectional file↔DB sync surface (`Notes/sync_engine.py` ties) and the file-notes workspace (`LibraryFileNotesWorkspace`, lazy-constructed) — workspace-owned pixels stay with the widget; the ≥2-subsystems rule governs the sync-status fields the shell also reads.
  - The 1 BLOCKED field from wave 7 (`_library_pending_list_entry_media_return`) is one of a FOUR-member shell family whose other members include the notes twin — the family stays screen-owned whole; do not move the notes member alone.
  - Two prefix families (`_library_note_*` and `_library_notes_*`) plus likely bare names (`_selected_note_id` per the conversations/skills/prompts/media precedent) — enumerate all.
  - Notes test coverage is deep and file-notes tests may live under additional roots — the ALL-of-Tests/ content-grep rule covers, but characterization must read existing coverage first.
- Split decision by connected-components evidence, media precedent (one controller unless a genuine second component); gate per wave 3 if a non-notes tangle emerges.

---

### Task 1: Notes state PR (series 1/N)
Ownership analysis (~138 flat fields; the sync/workspace seams and the four-member shell family ruled per field; package-qualified greps); characterization spot-check; `LibraryNotesState` (verbatim defaults/comments); programmatic shims with literal mapping assertions from birth; wiring test in the RED/pins commit; bypass seeds by content-grep INCLUDING shared-seam analysis. Fresh screen re-pin.

### Task 2: Notes controller move(s) (series 2/N)
Census (FIVE spellings + the on_<message> enumeration + call-graph); split decision with component evidence; RED wiring commit(s) → move commit(s) (byte-for-byte; born-lazy + preimport suffix; born-governed; constructor-binding pins + the both-directions AST resolver check; the canvas_sync notes branch + its guard) → blame-ignore (rev-parse, incl. state-PR). All guards green; fresh pins.

### Task 3: Notes cleanup (series 3/N)
Five-spelling census first + field-name prose sweep; retargets (boundary-matched tooling only); shim deletion; delegator pruning under the three-member whitelist; dead imports (exact-name `_SURFACE`); modal-inventory rows if presenters moved; recipe table; fresh pins.

### Task 4: Wave close — AND PROGRAM CLOSE PREP
Recipe trajectory + lessons; stale-doc sweep; durable evidence; full battery + paired sweeps (§7 net, isolated worktrees, parity-proven) + order-swapped probe; follow-up filings with id sweeps (collision risk is HIGH — dev mints ids hourly; sweep immediately before filing). Additionally: a program-close summary section in the recipe (§ trajectory across all eight waves, the census-spelling/bypass-shape catalogue as the reusable artifact, the phase-C handoff: probe baselines, the mount-storm evidence, TASK-31880 as the coverage gate).

## Self-review record
- The three wave-7 inheritances are pinned as Global Constraints with concrete actions, not warnings.
- The four-member shell family rule prevents the symmetric mistake of moving the notes twin alone.
- Phase C remains fenced out; task 4 explicitly preps its handoff.
- All mechanics by reference; only notes-specific decisions pinned here.
