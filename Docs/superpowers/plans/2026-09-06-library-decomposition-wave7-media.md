# Library Decomposition Wave 7 — Media Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the media subsystem from `LibraryScreen` — rough 2026-09-06 measure: ~244 media-named methods / ~122 distinct flat `_library_media_*` attribute names, by far the largest cluster of the program (prompts, the previous record, was 161 candidates / 46 fields). Notes (wave 8) remains after this; phase C (the resident-canvas click-freeze fix, the program's first motivated behavior change) follows wave 8 and is explicitly NOT part of this wave — wave 7 is pure moves only.

**Architecture:** Identical mechanics to the six merged series. `backlog/docs/library-decomposition-recipe.md` (all §, as updated by wave 6: the `on_<message>` name-dispatch whitelist member in §4, the patch-target-table and field-name-prose census shapes in §3, the unconditional isolated-worktree rule) is the how; this plan pins boundaries and wave-specific decisions. Templates: `library_prompts_controller.py`/`_state.py` (newest, largest prior move) for mechanics; wave 3's gated combined series for the split contingency.

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (on dev, as corrected).

## Global Constraints

- Everything prior waves' Global Constraints said, verbatim, plus all recipe lessons through wave 6 (RED wiring commit criterion; byte-for-byte canon; per-move pin lowering BOTH guard files; born-governed rows same-commit; the FOUR census spellings — attribute, quoted-string, bare-assignment, patch-target-table — plus the field-name prose sweep; the prune whitelist `@on` + `action_*` + `on_<message>` name-dispatch; content-grep across ALL of Tests/ for bypass fixtures; isolated worktrees with own venvs UNCONDITIONALLY for baselines; 10-run matched batches for timing-sensitive dispositions; probe order-swap; verify every number AND line range against live files before writing — this program's recurring failure class, three incidents in wave 6 alone; rev-parse-only hashes; erratum-not-amend for hash-load-bearing commits; backlog id sweep across origin/* + local before filing; never park silently; dev-race reconciliation with semantic ports, budgeted per wave).
- **Born-lazy controller import** (now guard-enforced: `Tests/Packaging/test_library_preimport_closure.py` lists the deferred suffixes — add the new controller's suffix in the move commit); state module import stays module-level.
- Known media-specific facts to respect:
  - `library_media_browse_controller.py` and `library_media_trash_browse_controller.py` (both existing in Library_Modules) are prior-extracted WIRING — untouched; their screen instances and the methods delegating to them are exclusion candidates (skills/prompts precedent).
  - **The `library_media_browse_controller` `_BUDGETS` row is a standing dev-owned red (410 lines vs 371 pin, creep predating the wave-6 merge-base; two waves have deliberately refused to re-pin it green).** This wave will work in that neighborhood: if any wave commit legitimately edits that file, re-measure and re-pin it in that commit with a comment attributing the creep window; otherwise leave it flagged — do NOT absorb it silently.
  - Media entangles with dev's TASK-31521 screen-reuse machinery (`_library_media_selection_timer`, `_stop_library_media_selection_debounce`, the suspend/resume gates, `_library_visit_entered` repeat-visit branches) — screen-lifecycle state is screen-owned (accessor bindings at most, canvas-resync precedent), never moved into media state; ownership analysis must rule each suspend/resume-adjacent field/method explicitly.
  - Media also owns reader/viewer surfaces that other subsystems' generic dispatchers touch (`_toggle_library_media_reader_pane` is the wave-6-verified generic dispatcher that reads FOUR state objects) — dispatcher stays screen-resident (exclusion), per the prompts-series ruling.
- **Split decision is REAL this wave.** At ~244 methods a single controller would be ~8-9k lines. Ownership analysis decides: (a) one controller (only if connected-components genuinely shows one blob); (b) two sequential move series within this wave (e.g. browse/viewer vs trash/maintenance) each with its own wiring pins, skills/prompts mechanics; (c) if entanglement analysis shows a search-RAG-grade tangle with a non-media cluster, STOP and gate per the wave-3 precedent. Record the decision with the component evidence.

---

### Task 1: Media state PR (series 1/N)
Ownership analysis (~122 flat fields; the suspend/resume seam ruled field-by-field; package-qualified greps); characterization spot-check (genuinely-unpressed handlers only — verify against existing coverage first; media has deep existing test files); `LibraryMediaState` (exclusive fields, verbatim defaults/comments); programmatic shims; wiring test in the RED/pins commit; bypass-fixture seeds by content-grep. Fresh screen re-pin.

### Task 2: Media controller move(s) (series 2/N)
Census (~244 candidates, four spellings + name-dispatch whitelist + call-graph for bare-named cluster members); the split decision with connected-components evidence; RED wiring commit(s) → move commit(s) (byte-for-byte; born-lazy import + preimport-closure suffix; born-governed row(s); constructor-binding surface pins per the wave-6 pattern) → blame-ignore (rev-parse, incl. state-PR). All guards green; fresh pins.

### Task 3: Media cleanup (series 3/N)
Four-spelling census first + field-name prose sweep; screen/test retargets; shim deletion at zero consumers; delegator pruning under the three-member whitelist; dead imports (after exact-name `_SURFACE` check); modal-inventory rows for moved presenters (the `_OwnerScope` pattern from wave 6 — construction-proof if the file is still the TASK-31815 blocked guard); recipe series-table; fresh pins.

### Task 4: Wave close
Recipe trajectory + lessons; stale-doc sweep; durable SDD evidence; full battery (8 wiring suites now, all guard surfaces, characterization, preflight, paired-baseline sweep per §7's defined net in isolated worktrees, order-swapped probe vs recorded band); follow-up filings with id sweeps.

## Self-review record
- The split decision and the suspend/resume seam are this wave's two highest-risk analyses; both get explicit per-item rulings with evidence.
- The standing media_browse red gets an explicit absorb-only-if-touched rule, preserving two waves of refusal precedent.
- Phase C is explicitly fenced out; the freeze fix lands as its own motivated change after wave 8.
- All mechanics by reference; only media-specific decisions pinned here.
