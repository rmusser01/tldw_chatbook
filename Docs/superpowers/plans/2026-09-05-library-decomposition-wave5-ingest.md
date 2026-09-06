# Library Decomposition Wave 5 — Ingest Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the ingest subsystem from `LibraryScreen` (78 ingest-named methods / 20 `__init__` fields at the 2026-09-05 measure) — the wave-4 scope ruling's deferred half.

**Architecture:** Identical mechanics to the five prior series. `backlog/docs/library-decomposition-recipe.md` (§1–§19, all lessons incl. the screen-identity bypass class) is the how; this plan pins ingest-specific decisions. Templates: `library_skills_controller.py`/`library_skills_state.py` (newest, incl. the three-prefix pattern if needed) + `Tests/Architecture/test_library_skills_wiring.py`.

**Spec:** `Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md` (dev, corrected).

## Global Constraints

- All prior waves' Global Constraints verbatim + all recipe lessons (RED-commit criterion; byte-for-byte canon; per-move pin lowering both guard files; born-governed same-commit; dynamic-dispatch census incl. dict.get→setattr; screen-identity bypass census incl. the "does the constant name resolve on the controller?" question; four test roots WHERE APPLICABLE — ingest's dedicated coverage lives mainly in Tests/UI but sweep Tests/Library + Tests/Live too, and check for a Tests/Ingest or similar; sequential paired-baseline sweeps; verbatim comments by copy-paste; rev-parse hashes; no-red-ships; verified-before-written claims; count accuracy; backlog id sweeps; never park; convergence loop at PR).
- Ingest-specific facts to respect:
  - **The ingest-options trio is permanently screen-routed** (`_INGEST_OPTIONS_CACHE_ATTR`, `_read_library_ingest_options_from_config`, `_library_ingest_options_for` — module-globals coupling, the FOUNDATION's original discovery, recipe §3's oldest entry). They are module-level in Library_Modules/screen_helpers or screen-resident — verify current location and DO NOT MOVE them.
  - Ingest has worker-heavy flows (upload/parse lanes, job history) — expect `@work` carriers and the ingest worker-group constants; enumerate early.
  - `_pause_library_ingest_transient_ui` and the ingest form-persistence behavior (task-2043: form survives rail switches) are shell-touched — ownership census decides, with the persistence contract explicitly protected by characterization if uncovered.
  - `Library_Ingest_Jobs_DB` interactions and the ingest rail width logic (`_sync_library_ingest_rail_for_width`) may be shell-owned — census decides.

---

### Task 1: Ingest state PR (series 1/3)
Ownership analysis (ingest prefix family; verify oddballs); characterization spot-check (genuinely-unpressed handlers; the form-persistence contract pinned if uncovered); `LibraryIngestState`; programmatic shims; wiring test in the RED/pins commit; fresh screen re-pin if lines move.

### Task 2: Ingest controller move (series 2/3)
RED wiring commit → move commit (byte-for-byte; four binding categories; ALL bypass censuses incl. screen-identity; free-name walk; born-governed row; single-vs-split by call-graph) → blame-ignore. Both guards + census green; fresh pins.

### Task 3: Ingest cleanup (series 3/3)
Dynamic-dispatch census first; retargets (screen + tests, assertions byte-for-byte); shim deletion at zero consumers; delegator census pruning; AST-derived dead-import prune; canon-scope docstring fixes; recipe series-table; fresh pins.

### Task 4: Wave close
Recipe trajectory + lessons; follow-up filings if any; stale-doc sweep; durable SDD evidence; full battery (six wiring suites after this wave, all characterization files, three ratchet surfaces, census, support-layer, preflight, sequential paired-baseline sweep, probe vs band with load noted).

## Self-review record
- The ingest-options trio is the wave's named no-touch hazard; stated twice.
- Form-persistence (task-2043) is the wave's behavioral contract to pin before moving.
- All mechanics by reference; only ingest decisions pinned here.
