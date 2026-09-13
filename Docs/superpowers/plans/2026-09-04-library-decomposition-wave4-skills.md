# Library Decomposition Wave 4 — Skills Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the skills subsystem from `LibraryScreen` — at 133 skill-named methods / 38 `__init__` fields (2026-09-04 measure, nearly double the spec's month-old snapshot), the largest single series of the effort. Ingest (78/20) is deliberately deferred to wave 5; one subsystem in flight at a time.

**Architecture:** Identical mechanics to the conversations/export/collections/search+RAG series. `backlog/docs/library-decomposition-recipe.md` (§1–§18 + all wave lessons) is the how; this plan pins boundaries and wave-specific decisions. Templates: `library_rag_search_controller.py`/`_state.py` (newest), collections series (largest prior single move).

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (on dev, as corrected).

## Global Constraints

- Everything prior waves' Global Constraints said, verbatim, plus all recipe lessons (RED wiring commit = screen untouched + pins red at parent; byte-for-byte canon; per-move pin lowering BOTH guard files; born-governed `_BUDGETS` row same-commit; dynamic-dispatch census incl. dict.get→setattr; three test roots; sequential paired-baseline sweeps; verbatim comments by copy-paste; rev-parse-only hashes; no-red-ships; verified-before-written evidence claims; backlog id sweep before filing; never park silently; dev-race convergence protocol at PR time).
- Known skills-specific facts to respect:
  - `library_skill_import_controller.py` (existing, primary class `LibrarySkillImportCoordinator`) and `library_skills_browse_controller.py` already live in Library_Modules — untouched; methods delegating to them are exclusion candidates (already-extracted wiring).
  - `_library_skill_import_coordinator` field on the screen is WIRING, stays (capture-controller precedent).
  - The skills cluster spans two prefix families (`_library_skill_*` and `_library_skills_*`) — enumerate both, no shortcuts; the trust/reset confirm-gate state (`_library_skill_trust_confirming_reset`) is touched by rail-switch shell code — ownership analysis decides move/stay per the ≥2-subsystems rule.
  - Skills has heavy Tests/Skills coverage OUTSIDE the usual three roots — sweep `Tests/Skills/` as a FOURTH root for bypass shapes and characterization checks.
- At 133 methods, the move may split into TWO controllers if ownership analysis shows a clean seam (e.g. editor/trust vs browse/list wiring); a split is sequential commits each with wiring pins; when unsure, one controller.

---

### Task 1: Skills state PR (series 1/3)
Ownership analysis (both prefix families + Tests/Skills root); characterization spot-check (genuinely-unpressed handlers only, four roots); `LibrarySkillsState` (exclusive fields, verbatim defaults/comments); programmatic screen shims; wiring test in the RED/pins commit. Fresh screen re-pin if lines move.

### Task 2: Skills controller move (series 2/3)
RED wiring commit → move commit(s) (byte-for-byte; four binding categories; @work + monkeypatch + module-globals + silent-Mock exclusion sweeps across FOUR roots; free-name walk; born-governed row; single-vs-split decision recorded) → blame-ignore. Both guards + census green; fresh pins.

### Task 3: Skills cleanup (series 3/3)
Dynamic-dispatch census first; screen retargets; test retargets (four roots, assertions byte-for-byte); shim deletion at zero consumers; delegator census pruning; dead-import prune; moved-docstring staleness fixes (canon ruling scope); recipe series-table; fresh pins both files.

### Task 4: Wave close
Recipe trajectory + lessons; stale-doc sweep; durable SDD evidence force-add; full battery (all five wiring suites now, characterization files, three ratchet surfaces, census, support-layer, preflight, sequential paired-baseline sweep, probe vs recorded band).

## Self-review record
- Single-subsystem scope is the honest reading of the 133-method measure; ingest deferred rather than rushed.
- The fourth test root (Tests/Skills) is this wave's most likely novel trap; named in every task.
- All mechanics by reference; only skills-specific decisions pinned here.
