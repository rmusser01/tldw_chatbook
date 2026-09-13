# Library Decomposition Wave 6 — Prompts Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the prompts subsystem from `LibraryScreen` — rough 2026-09-05 measure: ~152 prompt-named methods / ~65 distinct flat `_library_prompt[s]_*` attribute names, the largest single series of the effort (skills was 133). Media and notes remain for waves 7–8; one subsystem in flight at a time.

**Architecture:** Identical mechanics to the conversations/export/collections/search+RAG/skills/ingest series. `backlog/docs/library-decomposition-recipe.md` (all §, incl. the wave-5 seventh/eighth bypass shapes) is the how; this plan pins boundaries and wave-specific decisions. Templates: `library_ingest_controller.py`/`library_ingest_state.py` (newest, incl. the accessor-binding precedents), skills series (largest prior single move).

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (on dev, as corrected).

## Global Constraints

- Everything prior waves' Global Constraints said, verbatim, plus all recipe lessons (RED wiring commit = screen untouched + pins red at parent; byte-for-byte canon; per-move pin lowering BOTH guard files; born-governed `_BUDGETS` row same-commit; dynamic-dispatch census incl. dict.get→setattr AND quoted-string forms of every field name — the wave-5 round-2 string-loop incident; content-grep across ALL of `Tests/` for bypass fixtures, never `-k` filters; sequential paired-baseline sweeps in ISOLATED worktrees with own venvs; verbatim comments by copy-paste; rev-parse-only hashes; no-red-ships; verified-before-written evidence claims incl. NEVER stating an arity in a brief — measure it; backlog id sweep across origin/* + local before filing; never park silently; dev-race convergence protocol at PR time).
- **NEW, hard (wave-5 round-3 incident + dev's 06acf148f pre-import paydown):** the new controller's import into `library_screen.py` is **born function-local** in the established lazy-import block at the construction site — NEVER module-level. The state module import stays module-level (states are module-level on dev). `Tests/Packaging/test_library_preimport_closure.py` and the `_ui_ready` module census (pin 972) both enforce this; run both in every task's battery.
- Dev-race protocol (wave-5 evidence: three reconciliation rounds at 89/72/102 dev commits): the required check `Derived artifacts reproduce from their sources` now gates on a ~8-min "PR Fast Lane" (754 tests); convergence loops key on the required CHECK-RUN NAME, never workflow presence; catch-up merges are resolved semantically per the recipe's reconciliation entries (delegators stay; dev edits to moved bodies get PORTED into the controller behind accessor bindings; census re-run in attribute + string form after every merge). Budget for it.
- Known prompts-specific facts to respect:
  - `library_prompt_browse_controller.py` (existing, Library_Modules) is prior-extracted WIRING — untouched; screen methods delegating to it are exclusion candidates (skills-series precedent for browse/import coordinators).
  - The cluster spans two prefix families (`_library_prompt_*` and `_library_prompts_*`) — enumerate both, no shortcuts (skills-series precedent).
  - **Basename collision:** `tldw_chatbook/Library/library_prompts_state.py` (domain state, consumed by `_build_library_prompts_state`) already exists. The new UI state module `tldw_chatbook/UI/Library_Modules/library_prompts_state.py` will share its basename — the ingest wave shipped exactly this pair (`Library/library_ingest_state.py` vs `UI/Library_Modules/library_ingest_state.py`); every census/retarget grep must be package-qualified, and the two must never be confused in imports.
  - Extra test roots beyond UI/Library/Live: `Tests/Prompt_Management/`, `Tests/Prompt_Studio/`, `Tests/Internal_Prompts/`, `Tests/Prompts_DB/` — the all-of-`Tests/` content-grep rule already covers them, but characterization checks must LOOK in them for existing coverage before writing new pins.
- At ~152 methods, the move may split into TWO controllers if ownership analysis shows a clean seam (e.g. editor/studio vs browse/list wiring); a split is sequential commits each with wiring pins; when unsure, one controller.

---

### Task 1: Prompts state PR (series 1/3)
Ownership analysis (both prefix families; all test roots; package-qualified census against BOTH `library_prompts_state` modules); characterization spot-check (genuinely-unpressed handlers only); `LibraryPromptsState` (exclusive fields, verbatim defaults/comments); programmatic screen shims; wiring test in the RED/pins commit. Fresh screen re-pin if lines move.

### Task 2: Prompts controller move (series 2/3)
RED wiring commit → move commit(s) (byte-for-byte; four binding categories + accessor precedents from ingest; @work + monkeypatch + module-globals + silent-Mock + `object.__new__` fixture-seed sweeps across ALL of Tests/; free-name walk; born-governed row; born-lazy controller import; single-vs-split decision recorded) → blame-ignore (rev-parse; INCLUDING the state-PR commit — the gap fixed retroactively in wave 5). Both guards + census + preimport-closure + ui_ready census green; fresh pins.

### Task 3: Prompts cleanup (series 3/3)
Dynamic-dispatch census first (attribute + quoted-string forms); screen retargets; test retargets (assertions byte-for-byte); shim deletion at zero consumers; delegator census pruning; dead-import prune; moved-docstring staleness fixes (canon ruling scope); recipe series-table; fresh pins both files.

### Task 4: Wave close
Recipe trajectory + lessons (incl. any new bypass shapes); stale-doc sweep; durable SDD evidence force-add; full battery (all seven wiring suites now, characterization files, three ratchet surfaces + preimport-closure + ui_ready census, support-layer, preflight, sequential paired-baseline sweep vs the wave-6 start commit, probe vs recorded band).

## Self-review record
- Single-subsystem scope is the honest reading of the ~152-method measure; media/notes deferred rather than rushed.
- The born-lazy controller import and the basename collision are this wave's two most likely novel traps; both are named in Global Constraints and the relevant tasks.
- All mechanics by reference; only prompts-specific decisions pinned here.
